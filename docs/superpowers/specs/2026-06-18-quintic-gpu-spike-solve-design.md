# Design: Parallel banded solve for the quintic spline (GPU), shared-memory chunked / SPIKE

**Date:** 2026-06-18
**Branch:** `feat-quintic-spline`
**Status:** Design approved (brainstorm). Implementation is **GPU-gated** — correctness is
CPU-verifiable, but the speedup it targets cannot be measured without CUDA hardware.

---

## Context

`QuinticSplineInterpolant` (committed `ff9a92e`, relaid `7df6cf1`) builds each spline by
solving a banded B-spline collocation system. The GPU build/solve has three phases:
`fill_quintic_band` → `solve_quintic_band_batch` → `set_quintic_constants`. **Fill and
extract are already parallel over rows/segments** (millions of independent threads). The
**solve is one thread per spline** — a sequential de Boor banded LU.

The real consumer workload (LISA-like) is **few, enormous splines**: `ninterps ≈ 1–100`,
`length ≈ 5×10⁵ … ~1.9×10⁷` (1 month → ~3 yr at ~5 s cadence), grid **usually uniform**
(fixed cadence). In this regime the one-thread-per-spline solve uses ≤100 of ~100k GPU
threads while each grinds a multi-million-row sequential LU — effectively unusable. The
fill/extract phases are fine; **only the solve must be parallelized**, and shared memory
must be usable as *fixed-size buffers independent of `length`* so a single 19M-long spline
still parallelizes.

This supersedes the rationale behind backlog #1 (whole-band-in-shared-memory, length-capped
~1400) and #4 (`interp_i`-innermost layout, tuned for the *old* one-thread-per-spline solve).

## Goals / non-goals

**Goals**
- Parallelize the per-spline banded solve across the GPU with **intra-spline parallelism**,
  so `ninterps=1`, `length=19M` saturates the device.
- Use shared memory as **fixed-size chunk buffers**, independent of `length`.
- **Generalizable**: one algorithm; uniform grid is a fast specialization, non-uniform is the
  same kernels with per-chunk factors.
- **Exact / drop-in**: still reproduce `scipy.interpolate.make_interp_spline(x,y,k=5)` to
  ≤1e-13. Output `c1..c5` layout unchanged → no API/Python/binding/eval change.

**Non-goals (v1)**
- Fusing `fill` into the solve (keep materializing global band `W` for now).
- Changing the CPU path (it keeps today's sequential solve under `#else`).
- A log-grid Toeplitz path (log grids are non-uniform in `x` → general path).

## Approach: unified chunked / SPIKE-style banded solve

Replace **only** the GPU `solve_quintic_band_batch`. Partition each spline's `n` rows into
fixed-size chunks of `C` rows (`C` tuned so one chunk's band fits in shared memory; see
budget). The grid is **blocks over `(spline, chunk)` pairs** — total `Σ_s ⌈n_s/C⌉` blocks,
so the batch dimension is always "more blocks." This **unifies the regimes**:

| workload | chunks/spline | degenerates to |
|---|---|---|
| many short (e.g. 1000×2000, `n ≤ C`) | 1 | block-per-spline shared-mem solve (backlog #1); **no reduced system** |
| 100 × 0.5M | many | full SPIKE |
| 1 × 19M | ~`n/C` (~9300) | full SPIKE; saturates GPU at `ninterps=1` |

The collocation matrix is **totally positive** → **no pivoting** (same justification as the
current `quintic_banfac`). Bandwidth is `kl=ku=4` general, `2` (pentadiagonal) on uniform.

### Three kernels (replacing the single solve kernel)

1. **Local factor + reduce** — block per `(spline, chunk)`. Load the chunk's band rows
   (+ ≤4-row halo) into **shared memory**; run the local de Boor banded LU (`banfac`/partial
   `banslv`) on the chunk; compute the chunk's interface contributions ("spikes" — the
   bandwidth-≤4 coupling to the neighbouring chunks) and write them to a small **global
   reduced-system buffer**.
2. **Reduced solve** — per spline, a small block-tridiagonal system in the interface
   unknowns (size ≈ `4·(chunks−1)`). For few chunks: solve directly in one warp/block. For
   the `ninterps=1`, ~9300-chunk case the reduced system is itself large → **recurse with the
   same chunked solve (two-level SPIKE)**.
3. **Back-substitute** — block per `(spline, chunk)`. Reload the chunk into shared memory,
   substitute the now-known interface unknowns, finish the chunk's slice of the solution,
   write the B-spline coefficients to `B`.

### Shared-memory budget

Per chunk: band `9 × C × 8` bytes + rhs `C × 8` + small working space. For `C = 1024`:
`9·1024·8 = 73,728 B` + `8 KB` ≈ **~84 KB** → fits the ≥100 KB dynamic shared memory on
A100/H100 (opt-in via `cudaFuncAttributeMaxDynamicSharedMemorySize`). `C` is a tunable;
**independent of `length`** — this is the "buffers for long signals" property.

### Uniform fast path (primary case, `spline_type == LINEAR_SPACING`)

Interior chunks share an identical band (pentadiagonal Toeplitz, stencil
`[1,26,66,26,1]/120`). So:
- Factor **one** representative interior chunk; cache its LU factors (constant/shared) and
  **reuse for all interior chunks** — eliminates the dominant re-factoring cost.
- Interface spikes are then identical → the reduced system is **block-Toeplitz** → solvable
  by a constant-coefficient recurrence.
- Only the first/last chunks (carrying the special not-a-knot boundary rows, which can be
  bandwidth-4 there) are factored individually.

### Non-uniform handling + fallback

The same three kernels work non-uniform — each chunk simply factors its own band (no
caching). The current one-thread-per-spline solve is **retained as a compile/flag-guarded
safety fallback** (correctness net) until SPIKE is GPU-proven.

### Interaction with #4 (band layout)

#4 made `W` `interp_i`-innermost to coalesce the *old* one-thread-per-spline solve. SPIKE
instead wants a block to coalesce-load *one chunk of one spline*, which favours
**per-spline/per-chunk-contiguous** layout (≈ the original layout), **not** `interp_i`
innermost. Adopting SPIKE therefore **re-tunes (likely reverts) the #4 layout**. #4 stays
relevant only as the fallback solve's optimization. This is expected and documented, not a
regression.

## Data flow (GPU `interpolate_quintic`)

`fill_quintic_band` (unchanged) fills global band `W` and rhs `B` →
**[new] kernel 1 → kernel 2 → kernel 3** read/write `W`,`B`, reduced buffer →
`set_quintic_constants` (unchanged) reads solved `B`, writes `c1..c5`. `W` global footprint
is `9·n·8 ≈ 1.37 GB` at `n=19M` (fits for the small-`ninterps`-when-`n`-huge corner that GPU
memory forces anyway). Fusing it away is a documented follow-up.

## Edge cases

- `length < 6` — already guarded (raises). Quintic needs ≥ k+1 points.
- `n ≤ C` — exactly 1 chunk/spline → no reduced system (skip kernel 2); pure block-per-spline.
- `n` just over `C` (2 chunks); `n` not a multiple of `C` (ragged last chunk).
- Recursive reduced solve (`ninterps=1`, huge chunk count) — two-level SPIKE.
- Strictly-increasing `x` assumed (duplicate/clustered → singular, same as scipy/today).
- Reduced-system conditioning at very large chunk counts — mitigated by two-level recursion.

## Testing / verification

- **Correctness (CPU-verifiable, the gate):** implement the SPIKE logic so the same code
  compiles & runs serially on the CPU build, and validate against scipy `k=5` (≤1e-13) **and**
  against the current solve, across: uniform & non-uniform; `n < C`, `n = C+1`, `n` not a
  multiple of `C`, large `n`; `ninterps` 1 and >1 (exercise the batch + reduced paths). The
  method is direct/exact → results must match to FP, not just "close."
- **GPU speed (deferred to hardware):** `nsys`/`ncu` on the three kernels; confirm occupancy,
  coalesced chunk loads, shared-memory residency, and the `ninterps=1` long-single speedup vs
  the current solve. Cannot be measured on the dev Mac (no nvcc/GPU) — same gate as #4.
- **No-pivot stability:** justified by total positivity (as today); spot-check residual
  `‖A c − y‖` on stiff/non-uniform grids.

## Follow-ups (explicitly out of v1)

1. **Fuse fill into the solve** — compute each chunk's band on-the-fly into shared memory,
   never materialize global `W` (saves ~1.4 GB at `n=19M`; truest "shared-memory buffers").
2. **Tune `C`** and the direct-vs-recursive reduced-solve crossover per architecture.
3. **Truncated SPIKE** (drop fast-decaying spikes) if profiling shows the reduced solve
   dominates — viable because the matrix is diagonally well-behaved.
4. Revisit the `W` layout formally once SPIKE access patterns are profiled.

## References / precedent

- The cubic already ships an intra-spline parallel solve: `fit_cubic_spline_pcr`
  (parallel cyclic reduction, shared memory, `<<<1, threads, shared_bytes>>>`) — SPIKE is the
  banded generalization of that idea.
- SPIKE algorithm (Polizzi & Sameh) / block cyclic reduction for banded systems.
- Prior REJECTED idea (cuSPARSE `gpsvInterleavedBatch`) remains rejected for *this* purpose:
  it is a *batched* penta solver, useless at `ninterps=1` where intra-spline parallelism is
  the whole point.

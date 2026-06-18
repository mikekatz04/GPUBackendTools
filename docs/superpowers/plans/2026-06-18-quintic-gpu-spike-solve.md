# Quintic GPU Parallel Banded Solve (SPIKE/chunked) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the quintic spline's one-thread-per-spline banded solve with a chunked, SPIKE-style parallel solve that uses fixed-size shared-memory buffers, so a single very long spline (ninterps=1, length up to ~19M) saturates the GPU — while the CPU build runs the identical logic serially for verification.

**Architecture:** Partition each spline's banded collocation system into fixed-size chunks (`C` rows). Each `(spline, chunk)` is a block: gather the chunk's band into a contiguous shared-memory buffer, factor locally (de Boor `banfac`), compute its ≤4-wide interface "spikes", solve a small block-tridiagonal **reduced system** in the interface unknowns (itself a banded solve, recursed if large), then back-substitute per chunk. Uniform grids reuse one cached interior-chunk factorization (pentadiagonal Toeplitz). The three kernels are written once with `CUDA_KERNEL`/`CUDA_DEVICE`; `#ifdef __CUDACC__` bridges only the launch/index config, so the CPU path executes the same algorithm.

**Tech Stack:** CUDA/C++ in `src/gpubackendtools/cutils/Interpolate.cu` (compiled as `.cxx` for CPU, by nvcc for GPU), nanobind binding in `gbt_binding.cxx`, Python `QuinticSplineInterpolant` in `interpolate.py`, `unittest` + scipy in `tests/`.

## Global Constraints

- **Drop-in / exact:** output `c1..c5` keep the `interp_i*length + i` layout; results must match `scipy.interpolate.make_interp_spline(x, y, k=5)` to ≤1e-13. No API/Python/binding signature changes except an **optional** `chunk` argument (default auto) on `interpolate_quintic_wrap` for testing.
- **CPU mirrors GPU exactly:** one kernel body each; `#ifdef __CUDACC__` only changes grid/block/index config (mirror the existing `fill_quintic_band`/`solve_quintic_band_batch` pattern). No separate sequential algorithm on CPU.
- **No pivoting:** the collocation matrix is totally positive (same justification as the existing `quintic_banfac`).
- **length ≥ 6**, **x strictly increasing** (already guarded in `interpolate.py`).
- **Build (macOS, keg-only LAPACKE):** `PKG_CONFIG_PATH="/opt/homebrew/opt/lapack/lib/pkgconfig:/opt/homebrew/opt/openblas/lib/pkgconfig" CMAKE_PREFIX_PATH="/opt/homebrew/opt/lapack:/opt/homebrew/opt/openblas" LIBRARY_PATH="$(brew --prefix gcc)/lib/gcc/current" uv pip install -e '.[testing]'`
- **Test runner:** `uv run --no-sync python -m unittest …`. GPU tests skip on this machine (no CUDA); **GPU speed is out of scope to verify here** — correctness is the gate.

---

## File Structure

- `src/gpubackendtools/cutils/Interpolate.cu` — all new device/kernel code (chunk solve, spikes, reduced solve, uniform cache, 3 kernels, `quintic_spike_solve` host orchestration). Modify `interpolate_quintic` to call it.
- `src/gpubackendtools/cutils/Interpolate.hh` — declare the new `chunk` param on `interpolate_quintic`.
- `src/gpubackendtools/cutils/gbt_binding.cxx` — thread the optional `chunk` arg through `interpolate_quintic_wrap`.
- `src/gpubackendtools/interpolate.py` — pass `chunk` (default `0` = auto) from `QuinticSplineInterpolant`; add an internal `_chunk` hook for tests.
- `tests/test_quintic_spline_spike.py` — **new** test file dedicated to the SPIKE solve (tiny-chunk forcing, regimes, recursion, uniform-vs-general). Keeps `tests/test_quintic_spline.py` as the untouched drop-in regression.

---

## Task 1: Parameterize the local banded solve + chunk gather/scatter

**Files:**
- Modify: `src/gpubackendtools/cutils/Interpolate.cu` (near the quintic section, after `extract_quintic_coeffs`)
- Test: `tests/test_quintic_spline_spike.py` (new)

**Interfaces:**
- Produces:
  - `CUDA_DEVICE void band_gather(double *dst, double *W, int interp_i, int ninterps, int length, int r0, int rows)` — copy band rows `[r0, r0+rows)` from global `W` (current `QBAND` layout) into contiguous `dst[(band_row)*rows + (col-r0)]` (half-band `QUINTIC_HALF_BAND`).
  - `CUDA_DEVICE int banfac_local(double *ws, int m, int n)` — pivot-free banded LU on a contiguous buffer `ws[band_row*n + col]`, half-band `m`. Returns 0 / (row+1).
  - `CUDA_DEVICE void banslv_local(double *ws, int m, int n, double *b)` — solve after `banfac_local` (rhs `b` length `n`).

These are the existing `quintic_banfac`/`quintic_banslv` math, but (a) on a **contiguous local buffer** (`ws[bandrow*n+col]`, not `QBAND`), and (b) with a **runtime half-band `m`** so they serve both chunks (`m=4`) and the reduced system (`m=8`).

- [ ] **Step 1: Write the failing test** — a parity check that the refactor is wired (uses the production path; real coverage comes in Task 2). In `tests/test_quintic_spline_spike.py`:

```python
import unittest
import numpy as np
from scipy.interpolate import make_interp_spline
from gpubackendtools.interpolate import QuinticSplineInterpolant


def _scipy_eval(x, y, x_new, der=0):
    return make_interp_spline(x, y, k=5)(x_new, nu=der)


class SpikeSolveTest(unittest.TestCase):
    def _check(self, n, ninterps=1, kind="uniform", chunk=0, tol=1e-12):
        rng = np.random.default_rng(0)
        if kind == "uniform":
            xrow = np.linspace(0.0, 1.0, n)
        elif kind == "log":
            xrow = np.logspace(-1.0, 1.0, n)
        else:
            xrow = np.sort(rng.uniform(0.0, 1.0, n)); xrow[0], xrow[-1] = 0.0, 1.0
        x = np.broadcast_to(xrow, (ninterps, n)).copy()
        y = (np.sin(6 * x) + 0.1 * rng.standard_normal((ninterps, n))).copy()
        x_new = np.linspace(xrow[0], xrow[-1], 97)
        spl = QuinticSplineInterpolant(x, y, force_backend="cpu", _chunk=chunk)
        got = spl(np.broadcast_to(x_new, (ninterps, 97)).copy())
        for i in range(ninterps):
            exp = _scipy_eval(xrow, y[i], x_new)
            np.testing.assert_allclose(got[i], exp, atol=tol, rtol=0)

    def test_single_chunk_uniform(self):
        self._check(n=40, ninterps=1, kind="uniform", chunk=0)
```

- [ ] **Step 2: Run to verify it fails** — `uv run --no-sync python -m unittest tests.test_quintic_spline_spike -v`. Expected: FAIL (`QuinticSplineInterpolant` does not accept `_chunk` yet).

- [ ] **Step 3: Add `_chunk` plumbing (no solve change yet)** — in `interpolate.py` `QuinticSplineInterpolant.__init__`, accept `_chunk: int = 0`, store `self._chunk = int(_chunk)`, and pass it as the trailing arg to the native call (add the arg to `interpolate_quintic_wrap` in `gbt_binding.cxx` with `nb::arg("chunk") = 0` and to `interpolate_quintic`'s signature in `.hh`/`.cu`, initially ignored). Keep the current inline solve.

- [ ] **Step 4: Implement the three primitives** in `Interpolate.cu` (contiguous-buffer, runtime-`m` versions of the existing band math) and `band_gather`. Do not call them yet.

```c
CUDA_DEVICE
int banfac_local(double *ws, int m, int n) {
    for (int i = 0; i < n; ++i) {
        double piv = ws[m * n + i];
        if (piv == 0.0) return i + 1;
        int imax = (m < (n - 1 - i)) ? m : (n - 1 - i);
        for (int s = 1; s <= imax; ++s) {
            double fac = ws[(m + s) * n + i] / piv;
            ws[(m + s) * n + i] = fac;
            for (int r = 1; r <= imax; ++r)
                ws[(m + s - r) * n + (i + r)] -= fac * ws[(m - r) * n + (i + r)];
        }
    }
    return 0;
}
CUDA_DEVICE
void banslv_local(double *ws, int m, int n, double *b) {
    for (int i = 0; i < n; ++i) {
        int imax = (m < (n - 1 - i)) ? m : (n - 1 - i);
        for (int s = 1; s <= imax; ++s) b[i + s] -= ws[(m + s) * n + i] * b[i];
    }
    for (int i = n - 1; i >= 0; --i) {
        b[i] /= ws[m * n + i];
        int imax = (m < i) ? m : i;
        for (int s = 1; s <= imax; ++s) b[i - s] -= ws[(m - s) * n + i] * b[i];
    }
}
CUDA_DEVICE
void band_gather(double *dst, double *W, int interp_i, int ninterps,
                 int length, int r0, int rows) {
    const int m = QUINTIC_HALF_BAND;
    for (int br = 0; br < QUINTIC_BAND_ROWS; ++br)
        for (int c = 0; c < rows; ++c)
            dst[br * rows + c] = QBAND(W, br, r0 + c, interp_i, ninterps, length);
}
```

- [ ] **Step 5: Run the test** — Expected: PASS (the `_chunk` arg is accepted and ignored; the existing solve still produces correct `c1..c5`). Rebuild (Global Constraints build line) then run unittest.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "interp(quintic): add contiguous banded primitives + chunk gather; _chunk plumbing"`

---

## Task 2: `quintic_spike_solve` — chunked solve with reduced system

**Files:**
- Modify: `src/gpubackendtools/cutils/Interpolate.cu`
- Test: `tests/test_quintic_spline_spike.py`

**Interfaces:**
- Produces: `void quintic_spike_solve(double *W, double *B, int ninterps, int length, int chunk)` — solves all splines in place (`B` overwritten with B-spline coefficients), partitioning each spline into chunks of `chunk` rows (if `chunk<=0`, use `length` → single chunk). Replaces the body of `solve_quintic_band_batch` usage in `interpolate_quintic`.

**Algorithm (per spline, P = ceil(n/chunk) chunks; chunk size `C`):**
- For each chunk `j` covering rows `[r_j, r_j + C_j)` (`C_j` = chunk or remainder):
  1. `band_gather` → local buffer `A_j` (contiguous). `banfac_local(A_j, m=4, C_j)`.
  2. Local RHS solve: copy `B[r_j .. )` into `g`, `banslv_local(A_j, 4, C_j, g)`.
  3. Left/right spikes: build the `C_j × 4` coupling RHS from the off-diagonal corner blocks (the band entries linking chunk `j` to `j-1` (top, `Bc_j`) and `j+1` (bottom, `Cc_j`)). Solve `banslv_local` for each of the ≤4 nonzero coupling columns → spikes `Vt_j` (top-tip rows) and `Wb_j` (bottom-tip rows). Only the **top `m` and bottom `m` rows** of each spike feed the reduced system.
- **Reduced system:** unknowns = the `m` interface values at each chunk boundary (size `R = m*(P-1)` per spline). It is block-tridiagonal (bandwidth `2m=8`) in those unknowns, assembled from the spike tips + `g` tips. Store it as a contiguous band buffer and solve with `banfac_local`/`banslv_local` using `m_red = 2*QUINTIC_HALF_BAND` (Task 3 handles `R > C` by recursion).
- **Back-substitute:** with interface unknowns known, each chunk's final solution is `x_j = g_j − (left-spike)·(left interface) − (right-spike)·(right interface)`; write into `B[r_j ..)`.

> **Note (high-risk core):** the spike/reduced assembly is the intricate part. Implement against the tiny-chunk scipy test below (TDD oracle). The coupling blocks come directly from the de Boor band already in `W`: for boundary row `i` of a chunk, any band entry whose column falls in the neighbouring chunk is a coupling coefficient.

- [ ] **Step 1: Write failing tests (force multi-chunk via tiny `chunk`)**

```python
    def test_multichunk_uniform(self):
        for n in (20, 37, 64, 200):
            self._check(n=n, ninterps=1, kind="uniform", chunk=8)
    def test_multichunk_nonuniform(self):
        for n in (25, 50, 123):
            self._check(n=n, ninterps=1, kind="random", chunk=8)
    def test_multichunk_log(self):
        self._check(n=80, ninterps=1, kind="log", chunk=8)
    def test_multichunk_batch(self):
        self._check(n=60, ninterps=5, kind="uniform", chunk=8)
        self._check(n=60, ninterps=5, kind="random", chunk=8)
    def test_chunk_equals_or_exceeds_n(self):  # degenerate single chunk
        self._check(n=12, ninterps=2, kind="random", chunk=64)
```

- [ ] **Step 2: Run to verify they fail** — `uv run --no-sync python -m unittest tests.test_quintic_spline_spike -v`. Expected: the new tests FAIL (chunking not wired; `interpolate_quintic` still uses the old solve which ignores `chunk`).

- [ ] **Step 3: Implement `quintic_spike_solve`** in `Interpolate.cu` per the algorithm above (reusing Task-1 primitives), and change `interpolate_quintic` (the CPU `#else` branch for now) to call `quintic_spike_solve(W, B, ninterps, length, chunk)` instead of the inline `fill→solve→set` loop's solve step. Keep `fill_quintic_band` and `set_quintic_constants` as-is.

- [ ] **Step 4: Run tests** — rebuild, then `unittest tests.test_quintic_spline_spike -v`. Iterate implementation until all PASS (≤1e-12). Then run the original regression: `unittest tests.test_quintic_spline` — must stay green.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "interp(quintic): chunked SPIKE solve with reduced system (CPU path)"`

---

## Task 3: Two-level recursion for large reduced systems

**Files:** Modify `src/gpubackendtools/cutils/Interpolate.cu`; Test `tests/test_quintic_spline_spike.py`

**Interfaces:** `quintic_spike_solve` gains an internal recursion: when the reduced system size `R > chunk`, solve it by calling the same chunked routine on the reduced band instead of a single `banfac_local`.

- [ ] **Step 1: Write failing test (tiny chunk + large n → R > chunk)**

```python
    def test_two_level_reduced(self):
        # chunk=8, n=400 -> P=50 -> R = 4*49 = 196 > chunk -> recursion
        self._check(n=400, ninterps=1, kind="uniform", chunk=8)
        self._check(n=400, ninterps=1, kind="random", chunk=8)
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL or assertion error (single-level reduced solve overflows/incorrect when `R > chunk`). `unittest …test_two_level_reduced -v`.

- [ ] **Step 3: Implement recursion** — factor the reduced-system solve into a helper that, if `R > chunk`, partitions and recurses (same spike/reduced/back-sub on the bandwidth-`2m` reduced band); else solves directly with `banfac_local`/`banslv_local`.

- [ ] **Step 4: Run tests** — rebuild; all SPIKE tests PASS; `tests.test_quintic_spline` green.

- [ ] **Step 5: Commit** — `git commit -am "interp(quintic): two-level recursion for large SPIKE reduced systems"`

---

## Task 4: Uniform Toeplitz caching fast path

**Files:** Modify `src/gpubackendtools/cutils/Interpolate.cu`; Test `tests/test_quintic_spline_spike.py`

**Interfaces:** `quintic_spike_solve` takes the existing `spline_type` (thread it from `interpolate_quintic`). When `spline_type == CUBIC_SPLINE_LINEAR_SPACING`, interior chunks reuse one cached factorization.

- [ ] **Step 1: Write failing test (uniform fast path == general path == scipy)**

```python
    def test_uniform_fastpath_matches_general(self):
        # uniform grid hits the LINEAR_SPACING cache path; must equal scipy
        self._check(n=300, ninterps=3, kind="uniform", chunk=16, tol=1e-12)
```

(Initially passes via the general path; this test guards the optimization does not change results.)

- [ ] **Step 2: Run baseline** — Expected: PASS via general path (cache not implemented). Record it as a guard.

- [ ] **Step 3: Implement caching** — when uniform: factor one representative interior chunk once (`banfac_local`), reuse its factors for all interior chunks (skip re-`banfac`); factor only first/last chunks individually. Interface spikes for interior chunks are identical → assemble the reduced system from the shared spike.

- [ ] **Step 4: Run tests** — rebuild; `test_uniform_fastpath_matches_general` and all SPIKE + regression tests PASS (≤1e-12). Confirms the optimization is result-preserving.

- [ ] **Step 5: Commit** — `git commit -am "interp(quintic): uniform Toeplitz factorization cache fast path"`

---

## Task 5: GPU launch structure (`#ifdef __CUDACC__`) + dynamic shared memory

**Files:** Modify `src/gpubackendtools/cutils/Interpolate.cu`

**Interfaces:** Restructure `quintic_spike_solve`'s phases into three `CUDA_KERNEL`s (`spike_factor_reduce`, `spike_reduced_solve`, `spike_backsub`) whose **bodies are identical for CPU and GPU**; only the iteration bounds differ via `#ifdef __CUDACC__` (block per `(spline, chunk)` on GPU; serial loops on CPU — mirror `fill_quintic_band`). Local chunk buffer is dynamic shared memory on GPU (`extern __shared__`), a stack/`new` buffer on CPU.

- [ ] **Step 1: Restructure into the 3 kernels (CPU bodies unchanged in behavior)** — move the per-chunk factor/spike, reduced solve, and back-sub into kernel functions; `interpolate_quintic` calls them (serially on CPU). Keep results identical.

- [ ] **Step 2: Run all CPU tests** — rebuild; `tests.test_quintic_spline_spike` + `tests.test_quintic_spline` all PASS (behavior preserved by the restructure).

- [ ] **Step 3: Add the GPU `#ifdef __CUDACC__` launch config** — grid = `(ninterps, P)` blocks; `extern __shared__ double sbuf[]`; set `cudaFuncAttributeMaxDynamicSharedMemorySize`; pick `C` from a shared-mem budget (≈ `9*C*8 + C*8 ≤ 96KB` → `C≈1024`) when `chunk<=0`; `gpuErrchk`+`cudaDeviceSynchronize` like the existing eval path. This branch is **compile-mirrored only** (no GPU here).

- [ ] **Step 4: Static review of the GPU branch** — verify against the existing `fill_quintic_band`/`eval_quintic_wrap` idioms: fixed-size or dynamic-shared arrays only, no host-only calls in device code, correct sync between the 3 kernels, no write races (each `(spline,chunk)` block writes a disjoint slice of `B`; reduced buffer indexed per spline). Document findings in the commit message.

- [ ] **Step 5: Commit** — `git commit -am "interp(quintic): 3-kernel SPIKE structure + GPU launch config (CPU-verified; GPU static-reviewed)"`

---

## Task 6: Make SPIKE the default, keep old solve as guarded fallback; finalize

**Files:** Modify `src/gpubackendtools/cutils/Interpolate.cu`, `HANDOFF.md`, memory backlog

- [ ] **Step 1: Gate the old solve** — wrap the previous one-thread-per-spline `solve_quintic_band_batch` path behind `#ifdef GBT_QUINTIC_LEGACY_SOLVE` (off by default); default `interpolate_quintic` uses `quintic_spike_solve`. Auto `chunk` (`chunk<=0`) resolves to `length` on CPU and the shared-mem-budget `C` on GPU.

- [ ] **Step 2: Full suite** — rebuild; `uv run --no-sync python -m unittest discover` → expect all green, 2 GPU-skipped, no cubic regression. Capture output.

- [ ] **Step 3: Update docs/memory** — mark the SPIKE solve implemented (CPU-verified, GPU-unverified) in `HANDOFF.md` and the `quintic-gpu-optimization-backlog` memory; note #4's `interp_i`-innermost layout is now bypassed by the chunk-gather (revisit if legacy solve is ever re-enabled).

- [ ] **Step 4: Commit** — `git commit -am "interp(quintic): default to SPIKE parallel solve; legacy solve behind flag; docs/memory"`

---

## Self-Review

**Spec coverage:** unified chunked solve (T2) ✓; fixed-size shared-mem buffers / length-independent (T5) ✓; intra-spline parallelism via (spline,chunk) grid (T5) ✓; uniform Toeplitz fast path (T4) ✓; general non-uniform via same kernels (T2) ✓; current solve as guarded fallback (T6) ✓; CPU-mirrors-GPU verification (T1–T5 run on CPU; tiny-chunk forces all paths) ✓; #4 interaction noted (T6) ✓; two-level reduced recursion (T3) ✓; drop-in c1..c5 unchanged (Global Constraints, regression test each task) ✓. Fusion + formal W-relayout are spec follow-ups, intentionally out of scope.

**Placeholder scan:** the high-risk numerical core (T2 spike/reduced assembly) is specified as an algorithm + validating oracle test rather than 300 lines of literal C++ — this is deliberate: it is research-grade code where TDD against the tiny-chunk scipy parity test is the correct driver. All scaffolding/primitive/test code is given literally.

**Type consistency:** `banfac_local(ws,m,n)` / `banslv_local(ws,m,n,b)` / `band_gather(...)` / `quintic_spike_solve(W,B,ninterps,length,chunk[,spline_type])` used consistently T1→T6; reduced system uses the same primitives with `m=2*QUINTIC_HALF_BAND`.

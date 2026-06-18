# Handoff — Quintic (k=5) spline in GPUBackendTools, for downstream consumers

**Generated:** 2026-06-18 · **Updated:** 2026-06-18 (committed + GPU static-reviewed)
**Status:** 🟢 Ready for Next Step — CPU implementation complete & verified and
**now committed** (`ff9a92e` on `feat-quintic-spline`); GPU code written and
**statically reviewed** (no blocking issues — see below) but **not yet
nvcc-compiled/run**; branch **not merged**.

---

## Overview

`gpubackendtools` now provides a **quintic (degree-5) interpolating spline**,
`QuinticSplineInterpolant`, that is a **1-to-1 drop-in replacement** for the
existing `CubicSplineInterpolant` (identical constructor and `__call__`
signatures). It reproduces `scipy.interpolate.make_interp_spline(x, y, k=5)`
(default `bc_type=None`, not-a-knot) to **~1e-14** (machine precision). It
preserves the cubic's batching, the `interp_i*length + i` flat layout, the
spacing types, and is exposed in the native backends exactly like the cubic.

This handoff is for an agent in a **consumer repo** (LAT / GBGPU / BBHx / FEW /
similar) that wants to *use* the quintic — not continue building it. Lives on
branch **`feat-quintic-spline`** of `gpubackendtools` (repo:
`/Users/alessandrosantini/reps/globalift/erebor_org_setup/GPUBackendTools`).

---

## How to Use It (consumer API)

**Python — identical to the cubic, just swap the class name:**

```python
from gpubackendtools.interpolate import QuinticSplineInterpolant
import numpy as np

x = np.linspace(0.0, 1.0, 200)        # strictly increasing; length >= 6 REQUIRED
y = np.sin(6 * x)

# force_backend: "cpu" | "gpu" | "cuda" | "cuda12x" | ...  (default: first available)
spl = QuinticSplineInterpolant(x[None, :], y[None, :], force_backend="cpu")

y_new = spl(x_new[None, :])                 # evaluate
dydx  = spl(x_new[None, :], derivative=1)   # derivative = 0..5 (vs cubic's 0..3)
```

- **Batched:** pass shaped arrays `(..., length)` for `x`/`y_all`, or flattened
  arrays plus `ninterps=`, `length=`. Identical semantics to `CubicSplineInterpolant`.
- **Selective eval:** `spl(x_new, ind_interps=np.array([...]))` — same as cubic.
- **Coefficients:** `spl.c1 .. spl.c5` are the per-segment power-basis (Taylor)
  coefficients about each left node; `y0 = y`. Eval is
  `y0 + c1·dx + c2·dx² + c3·dx³ + c4·dx⁴ + c5·dx⁵`.

**Native C++/CUDA (if your `.cu` evaluates splines directly):**

```cpp
// Path from:  python -c "import gpubackendtools; print(gpubackendtools.get_include())"
//   -> /Users/.../GPUBackendTools/src/gpubackendtools/cutils
#include "InterpolateDevice.hh"   // header-only; now also defines:
//   QuinticSpline  (members x0,y0,c1,c2,c3,c4,c5,ninterps,length,spline_type)
//   QuinticSplineSegment  (Horner eval + eval_single..eval_quint_derivative)
// Same get_window/binary_search machinery as CubicSpline. No link against
// Interpolate.cu needed for evaluation (mirrors the cubic).
```

**Backend method handles** (on a `Globals().get_backend("gbt_cpu"|"gbt_cuda12x"|...)`):
`backend.interpolate_quintic_wrap(x, y, c1, c2, c3, c4, c5, length, ninterps)`
(fills c1..c5 in place), `backend.QuinticSplineWrap`, `backend.QuinticSpline`.

---

## Plan (feature status)

1. ✅ Python reference prototype, validated vs scipy `make_interp_spline(k=5)` to ≤1.2e-13.
2. ✅ Native C++/CUDA port (`Interpolate.cu` + `InterpolateDevice.hh`).
3. ✅ nanobind bindings (`gbt_binding.cxx/.hpp`).
4. ✅ Backend wiring (`cutils/__init__.py`) — cpu + cuda11x/12x/13x.
5. ✅ `QuinticSplineInterpolant` (`interpolate.py`).
6. ✅ Tests (`tests/test_quintic_spline.py`) + visual plots (`tests/plot_splines.py`).
7. ✅ CPU build + full suite green; ✅ **committed** (`ff9a92e`); ✅ **GPU static
   review** (no blockers); ⏳ **GPU build/run on real hardware**; ⏳ **merge to main**.

---

## What Has Been Done

- Implemented the quintic via faithful B-spline collocation: not-a-knot knots,
  (4,4)-banded collocation, de Boor's pivot-free banded LU (`banfac`/`banslv`,
  stable — collocation matrix is totally positive), then per-segment Taylor
  extraction via NURBS A2.3 (`DersBasisFuns`). Knots computed on the fly.
- GPU path mirrors the cubic's three-phase structure (`fill_quintic_band` →
  `solve_quintic_band_batch` → `set_quintic_constants`), all global-memory,
  grid-strided, **arbitrary length** (no shared-memory cap).
- Full suite: **17 tests, 15 pass + 2 GPU-skipped** (no GPU here), **no cubic regression**.
- Compiled CPU backend matches scipy `make_interp_spline(k=5)` to ~1e-14 across
  n=6…1000, equal/log/random grids; quintic == gbt-cubic on degree≤3 data.

---

## What Worked

- No-pivot banded solve matches scipy's pivoted `gbsv` to ≤2.8e-13 (totally-positive stability).
- Drop-in parity: identical constructor/`__call__` to the cubic; CPU verified.
- Visual checks (`tests/plot_splines.py`): overlays, residuals (~4e-16), derivatives.

---

## What Did Not Work / Ruled Out

- **cuSPARSE for the solve:** there is no batched *general-banded* solver
  (`gtsv`=tridiagonal, `gpsv`=pentadiagonal only). The quintic collocation is
  (4,4)-banded, so the solve is a custom one-thread-per-spline banded LU.
- **Pentadiagonal nodal reformulation + cuSPARSE `gpsvInterleavedBatch`:**
  rejected — the solve is *not* the bottleneck (de Boor assembly/extraction
  dominate), so it adds complexity without speedup. Do not re-investigate.

---

## SPIKE parallel banded solve (2026-06-18) — CPU done, GPU written (UNVERIFIED)

The quintic solve was reworked from one-thread-per-spline to a **chunked /
SPIKE-style parallel banded solve** for the consumer's real regime (few but very
long splines: `ninterps` 1–100, `length` up to ~19M). Spec + plan:
`docs/superpowers/specs/2026-06-18-quintic-gpu-spike-solve-design.md`,
`docs/superpowers/plans/2026-06-18-quintic-gpu-spike-solve.md`.

- **CPU path: implemented and verified.** `quintic_spike_solve` / `spike_solve_one`
  in `Interpolate.cu`: partition each spline into chunks (≥2m rows), per-chunk de
  Boor LU + left/right spikes, reduced system assembled in interleaved physical
  order and solved banded, two-level recursion for large reduced systems, and a
  uniform-grid Toeplitz factorization cache. Validated vs scipy `k=5` to ≤1e-12
  (new suite `tests/test_quintic_spline_spike.py`, tiny `chunk` forces every path)
  and the algorithm was first proven in a NumPy prototype vs `np.linalg.solve`
  (worst rel-err 1.7e-16).
- **GPU path: written but UNVERIFIED.** Three kernels (`spike_factor_kernel` /
  `spike_reduced_kernel` / `spike_backsub_kernel`) + `quintic_spike_solve_gpu`,
  global-memory (no shared memory yet), single-level reduced solve. Never
  nvcc-compiled or run (no GPU here). Static-reviewed (fixed a last-chunk slice
  sizing bug, `Cmax=2C`).
- **Default + fallback:** SPIKE is the default solve on both CPU and GPU. Build
  with `-DGBT_QUINTIC_LEGACY_SOLVE` to fall back to the one-thread-per-spline
  solve (useful to baseline the GPU SPIKE on hardware).
- **#4 interaction:** the chunk solve gathers each chunk into a contiguous buffer,
  so it does **not** use #4's `interp_i`-innermost `W` layout; #4 now only matters
  for the legacy fallback solve.
- **GPU follow-ups (not done):** shared-memory chunk fusion (kill the global `Dg`
  footprint), GPU two-level recursion for the reduced solve (the single-level
  reduced solve is serial-per-spline → a bottleneck at `ninterps=1` huge), uniform
  Toeplitz caching on GPU, and the `Cmax=2C` slice-size halving.
- **On a GPU box:** build a CUDA wheel and run `tests/test_quintic_spline_spike.py`
  + `tests/test_quintic_spline.py` on `gbt_cuda*`; if the GPU SPIKE misbehaves,
  rebuild with `-DGBT_QUINTIC_LEGACY_SOLVE` and compare.

## GPU Static Review (2026-06-18) — no blocking issues

A static review of the CUDA paths (`Interpolate.cu`, `InterpolateDevice.hh`,
bindings, backend wiring) was done before committing. Verdict: **ship-ready
pending hardware.** The CUDA path is a faithful mirror of the already-proven
cubic kernels. Verified:
- **Compile-safety (nvcc):** every GPU idiom (`std::ceil((n+NT-1)/NT)`,
  `gpuErrchk`, `cudaMalloc`/`cudaMemset`/`cudaMemcpy`H2D/`<<<>>>`/sync/`cudaFree`,
  `eval_quintic_kernel → eval_quintic_wrap`) is one the cubic already uses in the
  same file. All device arrays are fixed-size (`ders[6][6]`, `ndu[6][6]`,
  `a[2][6]` via `QUINTIC_DEG`) — no VLAs. `printf`/`floor`/`log10`/recursion are
  nvcc-supported and CUDA-vs-CPU branches are `#ifdef __CUDACC__`-guarded.
- **GPU/CPU name selection:** `QuinticSpline{,Wrap} → *GPU` uses the same
  `__CUDA_COMPILATION__ || __CUDACC__` macro as the cubic, and `__init__.py`
  loads the matching `*GPU` symbols for all 3 CUDA backends — so the
  host-compiled `gbt_binding.cxx` rides the exact mechanism the cubic GPU wheel
  already proves. Include chain intact (`Interpolate.hh:11 → InterpolateDevice.hh`).
- **Parallel correctness:** proved the `fill_quintic_band` slab-write map
  `(m+i−j)·length + j` is injective over `(i,j)` → no write race; `solve` is
  one disjoint slab+rhs per thread → no race; three phases separated by
  `cudaDeviceSynchronize`; `W` zeroed (`cudaMemset` GPU / `new double[…]()` CPU);
  de Boor band indexing checks out against standard (4,4) banded LU.
- **Non-blocking notes:** local-mem pressure in `quintic_ders_basis_funs`
  (backlog item 6); `<<<ninterps,…>>>` hits grid-x limit only for unrealistic
  `ninterps > 2³¹`. Residual risk is purely "never touched nvcc."

## Criticalities & Things to Keep in Mind

- **GPU is UNVERIFIED (statically reviewed, not run).** The CUDA kernels compiled
  only in their CPU (`#else`) form; they have never been through `nvcc` or run on
  a GPU. The static review above found no blockers, but treat GPU results as
  unproven until a `gbt_backend_cudaXXx` wheel is built and the GPU tests pass.
- **`length >= 6`** per spline (quintic needs k+1 points); **x strictly increasing**
  (duplicate/clustered x → singular / accuracy loss, same as scipy).
- **Backends:** native only — `gbt_cpu`, `gbt_cuda11x/12x/13x`. **No JAX** quintic
  yet (`gbt_jax` is deferred; `QuinticSplineInterpolant.supported_backends()` excludes it).
- **`cpp_class` pointer lifetime:** `spl.cpp_class` (the native `QuinticSplineWrap`)
  holds **raw pointers** into the interpolant's `x_flat/y_flat/c*_flat`. **Keep the
  `QuinticSplineInterpolant` object alive** for as long as you use `cpp_class`,
  or you get garbage/NaN (a temporary gets GC'd). Same caveat as the cubic.
- **Rebuild needs LAPACKE on macOS (keg-only Homebrew):**
  ```
  PKG_CONFIG_PATH="/opt/homebrew/opt/lapack/lib/pkgconfig:/opt/homebrew/opt/openblas/lib/pkgconfig" \
  CMAKE_PREFIX_PATH="/opt/homebrew/opt/lapack:/opt/homebrew/opt/openblas" \
  LIBRARY_PATH="$(brew --prefix gcc)/lib/gcc/current" \
  uv pip install -e '.[testing]'
  ```
  Project is **uv-managed** (venv one dir above the repo); use `uv run --no-sync python ...`.
- **Performance:** quintic fit ~18× the cubic on CPU; eval ~1.0–1.5×. *Estimated*
  A100: fit ~3–8×, eval ~1.3–1.7× (not measured). A GPU optimization backlog
  exists (top item: relayout the band `interp_i`-innermost to coalesce the solve).

---

## Next Steps

For the **consumer repo**:
1. Depend on `gpubackendtools` at branch `feat-quintic-spline` (until merged),
   e.g. install editable from the path above, or pin the branch in your env.
2. Swap `CubicSplineInterpolant` → `QuinticSplineInterpolant` where you want C⁴
   smoothness; the call sites need no other change (mind `length >= 6`).
3. If you evaluate splines in your own `.cu`: `#include "InterpolateDevice.hh"`
   and use `QuinticSpline` (path from `gpubackendtools.get_include()`).
4. **On a GPU box:** build a CUDA plugin wheel
   (`pip install --config-settings=cmake.define.GBT_WITH_GPU=ON .` or the
   `gpubackendtools-cudaXXx` build), then run
   `uv run python -m unittest tests.test_quintic_spline` and confirm `test_gpu`
   and the GPU `c_backend` test pass — this is the missing verification.

For `gpubackendtools` maintainers: merge `feat-quintic-spline` after GPU verify.

---

## Modified Files

| Status | Path | What changed |
|--------|------|--------------|
| [~] | `src/gpubackendtools/cutils/Interpolate.cu` | +quintic device primitives (de Boor basis/derivs, knots, no-pivot banded LU), 3 kernels, `interpolate_quintic`, `eval_quintic_wrap` |
| [~] | `src/gpubackendtools/cutils/Interpolate.hh` | declared `interpolate_quintic` + `eval_quintic_wrap` |
| [~] | `src/gpubackendtools/cutils/InterpolateDevice.hh` | +`QuinticSpline` / `QuinticSplineSegment` device classes (Horner eval, derivs 1–5); `#define QuinticSpline ...GPU/CPU` |
| [~] | `src/gpubackendtools/cutils/gbt_binding.cxx` | +`interpolate_quintic_wrap`, `QuinticSplineWrap::eval_wrap_func`, registered `QuinticSplineWrap{CPU,GPU}` + `QuinticSpline{CPU,GPU}`, `m.def` |
| [~] | `src/gpubackendtools/cutils/gbt_binding.hpp` | +`QuinticSplineWrap` wrapper class |
| [~] | `src/gpubackendtools/cutils/__init__.py` | +quintic fields on `GBTBackendMethods`/`GBTBackend`; wired into all 4 native loaders (JAX untouched) |
| [~] | `src/gpubackendtools/interpolate.py` | +`QuinticSplineInterpolant` (degree=5, c1..c5, length≥6, native-only backends, derivative 0–5, ninterps tiling) |
| [+] | `tests/test_quintic_spline.py` | scipy-k5 parity, cubic cross-check on degree≤3 data, single-spline fit, derivatives, length guard |
| [+] | `tests/plot_splines.py` | visual comparison plots (overlay+truth, residuals, derivatives); runnable + unittest |
| [+] | `tests/plots/*.png` | generated figures (overlay_behavior, agreement_residuals, derivatives) |

All nine source/test files above are **committed in `ff9a92e`**. Intentionally
left untracked: `HANDOFF.md` (this file; local abs paths), `tests/plots/*.png`
(regenerable via `tests/plot_splines.py`), and `src/gpubackendtools/_version.py`
(setuptools_scm artifact — not part of this work).

---

## Open Questions

- **Typical `ninterps` (batch width) in the consumer's workload?** Decides the
  GPU optimization path: large `ninterps` → the band-coalescing relayout alone
  suffices; few-but-very-long splines → would also need within-spline cooperative
  solve (harder, shared-memory tension). Not blocking — the current code is
  correct and arbitrary-length regardless.

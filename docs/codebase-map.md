_Last mapped: e42f469 · 2026-07-10 · regenerate when structure changes_

> Repo: `GPUBackendTools` (package import name `gpubackendtools`, current git branch `spline`).
> Role: **foundation library** — owns the compute-backend registry and shared C++/CUDA
> plumbing that LISAanalysistools (LAT), GBGPU, BBHx, FastEMRIWaveforms (FEW) and Eryn all sit on.

## 1. What this is

`gpubackendtools` is a hybrid Python/C++/CUDA package (built with `scikit-build-core` + CMake)
that provides (a) a generic **backend-selection framework** — abstract `Backend` classes, a
`BackendsManager` registry, a `Globals()` singleton, and a `ParallelModuleBase` mixin — that lets
any "compute object" pick CPU (NumPy), CUDA 11/12/13x (CuPy), or JAX at construction time and
dispatch consistently thereafter; and (b) a **reference implementation** of that framework: a
cubic-spline / interpolation kernel (`CubicSplineInterpolant` + native `Interpolate.{cu,hh}` +
nanobind bindings) that doubles as the template every downstream package copies to add its own
backends. It also ships the C++ headers shared across LISA Analysis Tools (`cuda_complex.hpp`, `gbt_global.h`,
`InterpolateDevice.hh`) that downstream native code `#include`s directly.

## 2. Layout

- `src/gpubackendtools/__init__.py` — registers the 4 GBT backends (`gbt_cpu`, `gbt_cuda{11,12,13}x`) + optional `gbt_jax`; exposes `get_include()` / `get_cmake_module_path()`.
- `src/gpubackendtools/gpubackendtools.py` — `Backend`, `BackendMethods`, `CpuBackend`, `Cuda{11,12,13}xBackend`, `_CudaBackend`, `BackendsManager`, `BackendStatus*`. The abstract machinery.
- `src/gpubackendtools/globals.py` — `Globals` singleton (config + logger + `BackendsManager`); module-level `get_backend`/`has_backend`/`get_first_backend`/`add_backends`/`initialize`/`reset`.
- `src/gpubackendtools/parallelbase.py` — `ParallelModuleBase` (generic) and `GBTParallelModuleBase` (prefixes `force_backend` with `"gbt_"`). The class downstream "compute object" classes subclass.
- `src/gpubackendtools/interpolate.py` — `CubicSplineInterpolant`, the reference `ParallelModuleBase` subclass; user-facing spline wrapper around the native backend.
- `src/gpubackendtools/pointeradjust.py` — `wrapper()`/`pointer_adjust` — legacy Cython pointer-marshalling helpers (numpy/cupy array → raw `size_t`).
- `src/gpubackendtools/cutils/` — native sources: `Interpolate.{cu,hh}`, `InterpolateDevice.hh`, `gbt_binding.{cxx,hpp}` (nanobind), `cuda_complex.hpp`, `gbt_global.h`, `cmake_functions.cmake`, `GPUBackendToolsConfig.cmake`, `CMakeLists.txt`. Also `cutils/__init__.py` — concrete `GBTCpuBackend`/`GBTCuda{11,12,13}xBackend` + `GBTBackendMethods` dataclass.
- `src/gpubackendtools/jax/` — `backend.py` (`GBTJaxBackend`), `cubic_spline.py` (pure-JAX spline fit/eval), gated on `import jax`; not yet documented in the repo's own `CLAUDE.md`.
- `src/gpubackendtools/utils/` — `config.py` (`Configuration`/`ConfigEntry`/env-var+CLI+file config plumbing, prefix `GPUBACKENDTOOLS_`), `citation.py`/`citations.py`, `utility.py`.
- `src/gpubackendtools/exceptions.py` — `GPUBACKENDTOOLSException` root + `BackendUnavailableException`/`MissingDependencies`/`MissingDriver`/etc. (also re-declared in `gpubackendtools.py` — prefer `.exceptions`).
- `CMakeLists.txt` (root) + `cmake/` — top-level CMake project; `GBT_WITH_GPU` (AUTO/ON/OFF/ONLY/BARE), `GBT_CUDA_ARCH`, LAPACKE detection knobs.
- `tests/test_cubic_spline.py` — the only test file; validates `CubicSplineInterpolant` against `scipy.interpolate.CubicSpline`.
- `docs/`, `examples/GPUBackendTools_tutorial.ipynb` — sphinx docs + tutorial notebook.

## 3. Core abstractions

```
Globals()  (singleton: logger, Configuration, BackendsManager)
   └── backends_manager: BackendsManager
          ├── _known_backends: {name -> Backend subclass}   (populated by add_backends())
          └── _registry: {name -> BackendStatus}             (Unloaded/Loaded/Disabled/Unavailable, lazy)

Backend (abstract)                     BackendMethods (dataclass: xp)
 ├── CpuBackend            (abstract)  → cpu_methods_loader()
 ├── _CudaBackend                       → check_cuda_backend() [module + cupy + driver-version checks]
 │    ├── Cuda11xBackend   (abstract)  → cuda11x_module_loader() / cuda11x_dynlib_loader()
 │    ├── Cuda12xBackend   (abstract)  → cuda12x_module_loader() / cuda12x_dynlib_loader()
 │    └── Cuda13xBackend   (abstract)  → cuda13x_module_loader() / cuda13x_dynlib_loader()
 └── (JAX subclasses Backend directly, not CpuBackend — no importlib module check)

Concrete GBT backends (src/gpubackendtools/cutils/__init__.py, jax/backend.py):
  GBTCpuBackend    _name="gbt_cpu"     _backend_name="gbt_backend_cpu"    xp=numpy
  GBTCuda11xBackend _name="gbt_cuda11x" _backend_name="gbt_backend_cuda11x" xp=cupy
  GBTCuda12xBackend _name="gbt_cuda12x" _backend_name="gbt_backend_cuda12x" xp=cupy
  GBTCuda13xBackend _name="gbt_cuda13x" _backend_name="gbt_backend_cuda13x" xp=cupy
  GBTJaxBackend     _name="gbt_jax"     _backend_name="gbt_backend_jax"     xp=jax.numpy (no compiled plugin — pure-Python)

ParallelModuleBase  (src/gpubackendtools/parallelbase.py)
  __init__(force_backend=None|str|Backend|(pkg,flavor))
  .backend -> Backend        .xp -> Backend.xp        .backend_name -> str
  .supported_backends()  [abstract; use CPU_ONLY/GPU_ONLY/GPU_RECOMMENDED/CPU_RECOMMENDED_WITH_GPU_SUPPORT]
  .build_with_same_backend(cls, args, kwargs)  /  .adapt_backend_kwargs(kwargs)
GBTParallelModuleBase(ParallelModuleBase)   # prefixes "gbt_" onto a bare force_backend string
```

- **`Backend.name`** is the *registry key* (`"gbt_cpu"`, `"gbt_cuda12x"`, ...) — what
  `get_backend()`/`BackendsManager` index by. **`Backend.backend_name`** is the *plugin module
  name* (`"gbt_backend_cpu"`) — what `_check_module_installed` imports. Don't confuse the two;
  `Backend.__reduce__` pickles by `.name`, not `.backend_name`.
- `BackendsManager.get_first_backend([...])` matches by **suffix after the first `_`** (so
  `"cuda12x"` matches registry key `"gbt_cuda12x"`), which is how the same generic list
  (`GPU_RECOMMENDED()` etc.) works identically for every downstream package's own `<pkg>_*`
  namespace.
- `get_backend("gbt_cuda")` / `has_backend("gbt_cuda")` special-case `"cuda"`/`"gpu"` tokens:
  they resolve to the first of `cuda13x → cuda12x → cuda11x` (see `globals.py:390-418`).

**Cubic-spline interpolation** (the reference backend implementation):
- `interpolate.py::CubicSplineInterpolant(GBTParallelModuleBase)` — builds `x/y/c1/c2/c3` flat
  arrays, dispatches spline *fit* to `self.backend.interpolate_wrap` (native) or a functional JAX
  path (`self.backend.name == "gbt_jax"` branch — C++ backends mutate buffers in place; JAX
  can't, so it returns coefficients instead), then evaluates splines in pure Python/NumPy/CuPy in
  `__call__` (binary-search segment lookup + cubic eval — this evaluation path does *not* call
  into the C++ `CubicSpline` class; it's a parallel NumPy-side implementation).
- `cutils/InterpolateDevice.hh` — **header-only** `CubicSpline`/`CubicSplineSegment` device
  classes (`eval`, `eval_single`, `get_window`, `binary_search`, `even_sampled_search`,
  derivatives). Downstream `.cu` files `#include` this directly to evaluate a spline without
  linking GBT's `Interpolate.cu` TU. Class name is aliased `CubicSpline` → `CubicSplineGPU` /
  `CubicSplineCPU` via `#define` at the top of the header (see §7 — CPU/GPU aliasing rule).
- `cutils/Interpolate.{cu,hh}` — the *build/solve* side: `interpolate()` (LAPACKE `dgtsv`
  tridiagonal solve, CPU), `fit_cubic_spline_thomas` (CPU-only Thomas algorithm),
  `fit_cubic_spline_pcr` (GPU-only Parallel Cyclic Reduction, one block/spline, shared-memory
  scratch). `Interpolate.cu` is **copy-compiled to `Interpolate.cxx`** at CMake time and compiled
  by both the CPU (g++) and GPU (nvcc) targets — do not edit the generated `.cxx`.
- `cutils/gbt_binding.{cxx,hpp}` — **nanobind** module (`NB_MODULE(interp, m)`) exposing
  `CubicSplineWrap` (aliased `CubicSplineWrapGPU`/`CubicSplineWrapCPU`), `CubicSpline`
  (`CubicSplineGPU`/`CubicSplineCPU`), `interpolate_wrap`, `fit_cubic_spline_thomas`/`_pcr`. One
  binding source compiles for both flavors via `#if defined(__CUDA_COMPILATION__)`. This is the
  **sole registrant** of `CubicSplineWrap`/`CubicSpline` — no other repo re-registers these types.
- `cutils/cuda_complex.hpp` — host/device-portable complex type; **single source shared across
  LISA Analysis Tools** (GBGPU/BBHx local duplicates were deleted at Phase 3.dedup; FEW's diverged
  copy is intentionally left alone — see MEMORY).
- `cutils/gbt_global.h` — `CUDA_CALLABLE_MEMBER`/`CUDA_DEVICE`/`CUDA_KERNEL`/`CUDA_SHARED`,
  per-axis `THREAD_START_{X,Y,Z}`/`BLOCK_INCR_{X,Y,Z}`/`BLOCK_START_{X,Y,Z}`/`GRID_INCR_{X,Y,Z}`
  macros (GPU: real CUDA intrinsics; CPU: degenerate to a single-iteration loop), `gpuErrchk`,
  and `typedef gcmplx::complex<double> cmplx`. Included transitively by nearly every kernel header
  across the LISA Analysis Tools repos.

## 4. Public API / entry points

Everything downstream imports is re-exported from `gpubackendtools/__init__.py`:

- `get_backend(name)` / `has_backend(name)` / `get_first_backend([...])` — module-level shortcuts
  onto `Globals().backends_manager`.
- `Globals()` — the singleton itself (`Globals().backends_manager.add_backends({...})` is how a
  downstream package registers its own backend classes at import time).
- `get_include() -> str` — absolute path to `cutils/` for `#include "gbt_global.h"` /
  `"cuda_complex.hpp"` / `"InterpolateDevice.hh"` from downstream C++/CUDA; shelled out to from
  CMake (`python -c "import gpubackendtools; print(gpubackendtools.get_include())"`).
- `get_cmake_module_path() -> str` — same directory, for `find_package(GPUBackendTools CONFIG)` →
  `GPUBackendTools::headers` interface target (via bundled `GPUBackendToolsConfig.cmake`); header-
  only, exports no compiled targets (every downstream recompiles its own TUs).
- `ParallelModuleBase` — the base class downstream "compute object" classes subclass.
- `wrapper()` / `pointer_adjust` — legacy Cython pointer-marshalling helpers (still used by some
  Cython-era entry points; the active native path is nanobind, not these).
- `interpolate.CubicSplineInterpolant` — concrete reference class (not re-exported at top level;
  `from gpubackendtools.interpolate import CubicSplineInterpolant`).

## 5. Backend-wheel model

- The **CPU backend ships in the main wheel** (`gpubackendtools`); each CUDA major version ships
  as a **separate plugin wheel** — `gpubackendtools-cuda11x`, `-cuda12x`, `-cuda13x` — controlled
  by the root `CMakeLists.txt` `GBT_WITH_GPU=ONLY` build mode. Each plugin wheel installs its
  compiled nanobind extension under a distinctly-named import package: `gbt_backend_cpu`,
  `gbt_backend_cuda11x`, `gbt_backend_cuda12x`, `gbt_backend_cuda13x` (target dir computed in
  `cmake/cmake_functions.cmake::apply_{cpu,gpu}_backend_common_options` as
  `${pkg_name}_backend_cpu` / `${pkg_name}_backend_cuda${CUDAToolkit_VERSION_MAJOR}x`, with
  `pkg_name=gbt`). The concrete backend classes (`GBTCuda12xBackend.cuda12x_module_loader`, etc.)
  `import gbt_backend_cuda12x.interp` lazily inside `check_cuda_backend()` — so a machine can have
  0, 1, or all 3 CUDA plugin wheels installed and only the ones actually importable become
  `BackendStatusLoaded`.
- **`Backend.__reduce__`** (`gpubackendtools.py:150-180`) makes every `Backend` instance picklable
  *by registry name* rather than by value: pickle stores `(_resolve_backend_for_unpickle, (name,))`
  and unpickling re-looks-up the singleton via `get_backend(name)` (importing the owning package
  first if needed, since the registry-name prefix — `gbt_`, `lisatools_`, ... — doubles as the
  import name). `__deepcopy__`/`__copy__` both `return self` for the same reason: a `Backend` is a
  process-wide singleton (it holds a live `xp` module reference and compiled-extension callables,
  neither of which is a meaningful "value" to duplicate). Committed 2026-07-10 (`bd6355c`,
  `e42f469`) — this is what lets `Backend` instances live safely inside settings trees that get
  `deepcopy`/`pickle`d (the LISA Analysis Tools–wide deepcopy/pickle-safety rule, see §7).
- **The `xp` property pattern**: `ParallelModuleBase.xp` returns `self.backend.xp` — never cache
  `self.xp = cp`/`self._xp = ...` as a plain attribute (that reintroduces the unpicklable-module
  problem `Backend.__reduce__` was built to solve one layer up). `CubicSplineInterpolant.xp` is
  itself just `self.backend.xp`.
- **Co-existence**: `has_backend("gbt_cpu")` and `has_backend("gbt_cuda12x")` can both be `True`
  simultaneously in the same interpreter because they're registered under distinct keys in one
  shared `BackendsManager._registry`, each lazily loaded from a distinctly-named Python package
  with distinctly-named C++ types (see §7 CPU/GPU aliasing) — no symbol or `typeid` collision
  between the CPU and GPU `.so`s even though both may be imported in the same process.

## 6. Cross-repo role

Every other LISA Analysis Tools repo depends on GBT. Verified via grep across sibling repos:

| Consumer | Python import | CMake |
|---|---|---|
| LISAanalysistools | `lisatools/__init__.py`, `response/tdionfly.py`, `cutils/__init__.py`, `utils/parallelbase.py`, `jax/backend.py`, `sources/emri/chebyshevwave.py` | `cutils/CMakeLists.txt` shells out to `gpubackendtools.get_include()` → `GBT_CUTILS`, checks `${GBT_CUTILS}/gbt_global.h` exists, adds to `target_include_directories` |
| GBGPU | `gbgpu.py`, `__init__.py`, `cutils/__init__.py`, `parallelbase.py` | `cutils/CMakeLists.txt` (same pattern) |
| BBHx | `__init__.py`, `waveformbuild.py`, `cutils/__init__.py`, `utils/parallelbase.py` | `cutils/CMakeLists.txt` (same pattern) |
| FastEMRIWaveforms | `cutils/__init__.py`, `__init__.py`, `utils/utility.py`, `utils/baseclasses.py` | not checked for CMake usage; **not declared in `pyproject.toml` dependencies** despite being imported (possible doc/packaging gap, not fixed here) |
| Eryn | `paraensemble.py` | pulled in via `pyproject.toml` dependency (no native build) |

Mechanism: (1) **Python** — every downstream `ParallelModuleBase`-style base class
(`lisatools.utils.parallelbase`, `gbgpu.parallelbase`, `bbhx.utils.parallelbase`, ...) subclasses
GBT's `ParallelModuleBase`/pattern and registers its own backends onto the *same*
`Globals().backends_manager` singleton via `add_backends({"<pkg>_cpu": ..., ...})`. (2) **CMake** —
every downstream native build shells out to `gpubackendtools.get_include()` (assigned to
`GBT_CUTILS` or similar) and adds it to `target_include_directories`, then `#include`s
`gbt_global.h`/`cuda_complex.hpp`/`InterpolateDevice.hh` directly; **no compiled library is
linked** — GBT ships headers only for this purpose (downstream `Interpolate.cu`/CubicSpline
consumers copy-compile the same pattern GBT itself uses, or link against GBT's Python-importable
`interp` module only for the spline fit, not for device-side eval).

## 7. Non-obvious invariants / gotchas

- **Single-registrant rule (L2, LISA Analysis Tools–wide)**: `CubicSplineWrap`/`CubicSpline` (and by extension
  any nanobind-registered type) must be registered by exactly one binding TU per backend flavor.
  GBT's `gbt_binding.cxx` is the sole registrant for these two types; the grep gate shared across
  LISA Analysis Tools (`tools/check_single_registrant.sh`, lives at the **umbrella workspace root**, not in this repo) enforces it
  elsewhere.
- **CPU/GPU class-name aliasing**: every type that ends up in both the CPU and GPU `.so` (wrapper
  classes *and* the underlying C++ classes) must be `#define`d to a CPU/GPU-suffixed alias in both
  branches of an `#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)` block — see
  `gbt_binding.hpp:39-43` (`CubicSplineWrap`) and `InterpolateDevice.hh:13-17` (`CubicSpline`).
  Missing this on one branch produces a backend-asymmetric class name and silent `typeid`
  collisions between the two plugin wheels.
- **`Interpolate.cu` → `Interpolate.cxx` copy is implicit** (`cutils/CMakeLists.txt:38-45`):
  editing `Interpolate.cu` triggers rebuilds of *both* CPU and GPU targets; never edit the
  generated `.cxx` directly.
- **`nanobind==2.12.0` pinned exactly** in `pyproject.toml` — "cross-wheel nanobind type sharing
  (OrbitsWrap, CubicSplineWrap, ...) requires every separately-built wheel to embed the same
  nanobind internals ABI. Bump in ONE sprint-wide PR only." (The umbrella workspace root's L1
  pybind11 pin documented in the root `CLAUDE.md`/MEMORY is **stale** post-Phase-3M — this repo,
  and the project as a whole, migrated pybind11 → nanobind; GBT's own `CLAUDE.md` still says "pybind11" throughout and
  should be treated as stale on that point too. Verify against `gbt_binding.cxx`/`.hpp`, which are
  unambiguously nanobind, `#include <nanobind/...>`, `NB_MODULE`.)
- **Deepcopy/pickle safety**: don't cache `self.xp = cp` as a plain attribute anywhere in an
  object that might enter a settings tree; use the `xp` *property* pattern (§5). `Backend`
  instances are deepcopy/pickle-safe via `__reduce__`/`__deepcopy__`/`__copy__` as of 2026-07-07
  (`bd6355c`) / 2026-07-10 (`e42f469`) — this repo is the origin of that LISA Analysis Tools–wide rule.
- **`cuda_complex.hpp` sole-source**: GBT is the single source shared across LISA Analysis Tools; GBGPU/BBHx local
  duplicates were deleted at Phase 3.dedup (2026-06-04); FEW's `std::`-qualified diverged copy is
  intentionally untouched.
- **`Backend.name` vs `Backend.backend_name`**: registry key vs. plugin-module name — easy to
  transpose; `Backend.__reduce__` explicitly documents this trap in its docstring.
- **JAX backend (`gbt_jax`) is undocumented in the repo's own `CLAUDE.md`** (added by commit
  `5622132`, "jax backend added" — no accompanying `CLAUDE.md` update found). It has no compiled
  plugin module (`_check_module_installed` is overridden to a no-op); `interpolate.py` special-
  cases it (`self.backend.name == "gbt_jax"`) because the JAX fit path is functional (returns
  coefficients) rather than in-place-mutating (like the C++ backends).
- **`git status` on this repo is currently on branch `spline`**, not `main`, with an untracked
  `_version.py` (normal setuptools_scm artifact, not committed).

## 8. Where to look for X

| Want to... | Start in |
|---|---|
| Understand backend selection / registry mechanics | `src/gpubackendtools/gpubackendtools.py` (`Backend`, `BackendsManager`) |
| Understand the `Globals()` singleton / config / `get_backend` | `src/gpubackendtools/globals.py` |
| Add a `ParallelModuleBase` subclass in a downstream package | `src/gpubackendtools/parallelbase.py` (copy the `GBTParallelModuleBase` pattern) |
| Add a new native (C++/CUDA) backend method | `cutils/Interpolate.{cu,hh}` (impl) → `cutils/gbt_binding.cxx`/`.hpp` (nanobind) → `cutils/__init__.py` (`GBTBackendMethods` field + wire into each `GBT*Backend`) |
| Evaluate a spline from device code without linking GBT | `cutils/InterpolateDevice.hh` (`#include` it directly) |
| Fix/extend spline fit-and-solve (LAPACKE / PCR) | `cutils/Interpolate.cu` + `cutils/Interpolate.hh` |
| Debug nanobind binding / module naming | `cutils/gbt_binding.cxx` (`NB_MODULE(interp, m)`), `cutils/gbt_binding.hpp` (`array_type<T>`, `CubicSplineWrap`) |
| Trace CMake plugin-wheel layout (`gbt_backend_cuda12x` etc.) | `cutils/CMakeLists.txt` + `cmake/cmake_functions.cmake` (`apply_{cpu,gpu}_backend_common_options`) |
| Change LAPACKE detection/fetch behavior | `cmake/cmake_functions.cmake` (`get_lapacke`, `try_get_lapacke_with_{pkgconfig,cmake,cpm}`) |
| Add/inspect the pure-JAX backend | `src/gpubackendtools/jax/backend.py` (`GBTJaxBackend`), `jax/cubic_spline.py` |
| Understand pickling/deepcopy of `Backend` | `src/gpubackendtools/gpubackendtools.py:150-180` (`Backend.__reduce__`, `_resolve_backend_for_unpickle`) |
| Understand `CubicSplineInterpolant` usage/shape conventions | `src/gpubackendtools/interpolate.py`; `tests/test_cubic_spline.py` for worked examples |
| Understand the GPU macros shared across LISA Analysis Tools (`CUDA_SHARED`, `THREAD_START_X`, ...) | `cutils/gbt_global.h` |
| See how a downstream repo consumes GBT headers via CMake | `LISAanalysistools/src/lisatools/cutils/CMakeLists.txt` lines ~20-55 (`GBT_CUTILS` shell-out) |
| Configuration / env vars (`GPUBACKENDTOOLS_*` prefix) | `src/gpubackendtools/utils/config.py` |

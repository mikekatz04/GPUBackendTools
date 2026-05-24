# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`gpubackendtools` (imported as `gpubackendtools`) is the **backend dispatch framework** that downstream LISA/GW packages (e.g. `lisaanalysistools`, `fastemriwaveforms`) use to select between CPU (NumPy) and CUDA (CuPy) implementations of native code at runtime. It is itself a hybrid Python / C++ / CUDA project built with `scikit-build-core` and CMake. Python ≥ 3.10 is required.

The package owns two responsibilities:
1. **Backend selection / loading machinery** — abstract `Backend`, `BackendsManager`, `Globals` singleton, the `ParallelModuleBase` class downstream code subclasses, and helpers (`get_backend`, `has_backend`, `pointer_adjust`, `wrapper`).
2. **Reference backend implementations** — a CPU and CUDA cubic-spline / interpolation kernel exposed as the concrete `GBT*Backend` classes used by the test suite and as a template for downstream packages.

## Build & Install

The project uses `scikit-build-core` (configured in `pyproject.toml`) and CMake (`CMakeLists.txt`) — **do not use `setup.py` directly**.

```sh
# Editable dev install (CPU; auto-detects CUDA toolkit if present)
pip install -e '.[testing]'

# From-source install (release)
pip install .
```

Build behavior is controlled by the CMake option `GBT_WITH_GPU` (`AUTO` / `ON` / `OFF` / `ONLY` / `BARE`). On `AUTO` (default), the GPU backend is built only if `nvcc` and the CUDA toolkit are found. CUDA architectures are controlled by `GBT_CUDA_ARCH` (default `native`). Override via `pip install --config-settings=cmake.define.GBT_WITH_GPU=ON .` etc.

LAPACKE is required for the CPU backend. `GBT_LAPACKE_DETECT_WITH` (`AUTO`/`CMAKE`/`PKGCONFIG`/`DISABLE`) chooses how to find a system install; `GBT_LAPACKE_FETCH` (`AUTO`/`ON`/`OFF`) controls whether sources are downloaded and built. `GBT_LAPACKE_EXTRA_LIBS` defaults to `gfortran` when linkable — set explicitly if linker complains about missing Fortran symbols.

CUDA-enabled builds are published as separate plugin wheels: `gpubackendtools-cuda11x`, `-cuda12x`, `-cuda13x`. Each ships a `gbt_backend_cudaXXx` Python module that the main package imports lazily.

## Testing

Tests live in `tests/` and use `unittest`:

```sh
# Run the full suite
python -m unittest discover

# Run a single test file / case / method
python -m unittest tests.test_cubic_spline
python -m unittest tests.test_cubic_spline.CubicSplineTest
python -m unittest tests.test_cubic_spline.CubicSplineTest.test_cubic_spline
```

The `[testing]` extras (`matplotlib`) must be installed.

## Lint / Format

Pre-commit is configured with `ruff` (lint + format), `cmake-format`/`cmake-lint`, `pyproject-fmt`, `mypy`, and `clang-format` (Google style for C/C++/CUDA).

```sh
pre-commit install         # one-time
pre-commit run --all-files # run on the whole tree
```

`ruff` is on a **white-list** approach (see `pyproject.toml [tool.ruff]`) — most files are excluded; only a handful under `src/gpubackendtools/files/` and `src/gpubackendtools/utils/` are linted. `T201` ("no `print`") is enforced where ruff runs.

## Backend Architecture

This is the most important concept in the codebase — the design downstream packages mirror.

A **backend** = `(xp, native_methods)` where `xp` is `numpy` or `cupy` and `native_methods` is a dataclass of C++/CUDA callables loaded from a per-CUDA-version plugin module (`gbt_backend_cpu`, `gbt_backend_cuda11x`, `gbt_backend_cuda12x`, `gbt_backend_cuda13x`).

- **Class hierarchy** (`src/gpubackendtools/gpubackendtools.py`):
  - `Backend` (abstract) → `CpuBackend`, `Cuda11xBackend`, `Cuda12xBackend`, `Cuda13xBackend`. The CUDA classes share `_CudaBackend` which performs CuPy/CUDA-version checks and (on Linux) attempts to dynamically load `nvidia-*` shared libs (`libcudart`, `libcublas`, `libnvJitLink`, `libcusparse`, `libnvrtc`) from the `nvidia` Python wheels.
  - `BackendMethods` (dataclass) — the contract for what a backend exposes; just `xp` at the base.
  - `BackendsManager` — a registry mapping name → `BackendStatus{Unloaded, Loaded, Disabled, Unavailable}`. Lazily constructs backends on first access and caches the result.
- **Globals** (`globals.py`) — `Globals()` is a singleton holding the `BackendsManager`, logger, and `Configuration`. `get_backend(name)` / `has_backend(name)` / `get_first_backend([...])` are module-level shortcuts onto it.
- **Name aliasing**: `get_backend("...cuda")` resolves to the first available of `cuda13x`/`cuda12x`/`cuda11x`; `"...gpu"` aliases `"...cuda"`. `BackendsManager.get_first_backend` matches on the suffix after `_` (e.g. `gbt_cuda12x` matches request `cuda12x`), enabling per-package namespacing (`gbt_*`, future `lisatools_*`, etc.).

### Adding a new backend method

To expose a new C++/CUDA function on the GBT backend:

1. Implement it in `src/gpubackendtools/cutils/Interpolate.cu` (or a new `.cu`/`.hh` pair) and bind it in `gbt_binding.cxx` using pybind11.
2. Add a field to `GBTBackendMethods` in `src/gpubackendtools/cutils/__init__.py`.
3. Wire the field into `GBTBackend.__init__` and into each concrete `GBT{Cpu,Cuda11x,Cuda12x,Cuda13x}Backend.*_module_loader()` so it is loaded from the right plugin module.

### Subclassing for downstream packages

Downstream packages register backends by extending `Backend`/`BackendMethods` analogously and adding them to the global manager:

```python
from gpubackendtools import Globals
Globals().backends_manager.add_backends({"mypkg_cpu": MyPkgCpuBackend, ...})
```

User-facing classes inherit from `ParallelModuleBase` (`parallelbase.py`), which exposes `self.backend`, `self.xp`, `self.backend_name`, and `build_with_same_backend(...)`. Subclasses must implement `supported_backends()` (use the `CPU_ONLY`, `GPU_ONLY`, `GPU_RECOMMENDED`, `CPU_RECOMMENDED_WITH_GPU_SUPPORT` static helpers). `GBTParallelModuleBase` is the GBT-namespaced variant that prefixes `gbt_` onto a string `force_backend`.

## Native Sources (`src/gpubackendtools/cutils/`)

- `Interpolate.cu` / `Interpolate.hh` — cubic-spline / general interpolation kernels. The `.cu` file is **copied to `Interpolate.cxx` at build time** and compiled by the C++ compiler for the CPU backend; the same `.cu` is compiled by `nvcc` for the GPU backend (see the `add_custom_command` in `cutils/CMakeLists.txt`). **Code must be valid as both** — guard CUDA-only intrinsics with the macros in `gbt_global.h`.
- `gbt_binding.cxx` / `gbt_binding.hpp` — pybind11 module exposing the C++/CUDA functions to Python (one shared binding source for both CPU and GPU builds; the produced extension is `gbt_backend_<flavor>.interp`).
- `pybind11_cuda_array_interface.hpp` — pybind11 caster that lets functions accept CuPy arrays via `__cuda_array_interface__`.
- `cuda_complex.hpp` — host/device-portable complex type.
- `CubicSpline.cu`, `interp.pyx`, `interp.pxd`, `dev_ptr_issue.pyx` — legacy Cython artifacts. The active path is the pybind11 module produced from `gbt_binding.cxx`. Cython sources are excluded from wheels (`wheel.exclude` in `pyproject.toml`).
- `cmake_functions.cmake` — `apply_cpu_backend_common_options` / `apply_gpu_backend_common_options` helpers and `get_lapacke()` LAPACKE detector.

## Python package layout (`src/gpubackendtools/`)

- `__init__.py` — registers the four `GBT*Backend` classes on `Globals().backends_manager` at import.
- `gpubackendtools.py` — `Backend`, `BackendMethods`, `BackendsManager`, all the CUDA-version-checking and `nvidia-*` dynlib-loading logic, plus `BackendStatus*` records.
- `globals.py` — `Globals` singleton, `Configuration` plumbing, and the `get_backend` / `has_backend` / `get_first_backend` / `initialize` / `reset` module-level API.
- `parallelbase.py` — `ParallelModuleBase` (the base class downstream "compute object" classes inherit from) and `GBTParallelModuleBase` (the GBT-flavored namespace variant).
- `interpolate.py` — `CubicSplineInterpolant`, the reference user-facing class wrapping the native `interpolate_wrap` / `CubicSplineWrap` / `CubicSpline` exposed via `GBTBackendMethods`.
- `pointeradjust.py` — `wrapper(...)` and `pointer_adjust` decorator: convert NumPy/CuPy arrays and Cython-class objects to raw `size_t` pointers for legacy Cython entry points. Used by `interpolate.py`.
- `cutils/__init__.py` — concrete `GBTCpuBackend` / `GBTCuda11xBackend` / `GBTCuda12xBackend` / `GBTCuda13xBackend` and the `GBTBackendMethods` dataclass.
- `exceptions.py` — `GPUBACKENDTOOLSException` root and `BackendUnavailableException` / `MissingDependencies` / `MissingDriver` / `BackendAccessException` etc. (Note: `BackendUnavailableException` is also re-defined in `gpubackendtools.py` for historical reasons — prefer importing from `.exceptions`.)
- `utils/` — `config.py` (Configuration / ConfigConsumer / `detect_cfg_file` for CLI/env/file-based config), `citation.py` / `citations.py` (pydantic-based citation metadata), `utility.py`.

## Key External Dependencies

- `pybind11`, `cython`, `numpy`, `scikit-build-core` — build-time.
- `nvidia-ml-py` (`pynvml`) — runtime CUDA driver-version detection in `_CudaBackend._get_cuda_version`.
- `pydantic`, `jsonschema`, `pyyaml` — used by the file/registry/citations layer.
- `cupy` is **not** declared in `dependencies` — it must be installed separately matching the chosen CUDA backend (`cupy-cuda12x` etc.). The CUDA backend constructors raise `MissingDependencies` listing the right `cupy-cudaXXx` and `nvidia-*-cuXX` pip packages when something is missing.
- `lapacke` (and `gfortran`) — system requirement for the CPU backend, fetched and built if not found (controlled by `GBT_LAPACKE_FETCH`).

## Notes for Working in This Tree

- The **`.cu` → `.cxx` copy** rebuild step is implicit: editing `Interpolate.cu` triggers rebuilds of both CPU and GPU targets. Do not edit the generated `Interpolate.cxx` directly.
- When adding Python code that should work on both CPU and GPU, take an `xp` (or use `self.xp` from `ParallelModuleBase`) rather than importing `cupy` unconditionally. The `try: import cupy as xp / except: import numpy as xp` fallback pattern in `pointeradjust.py` is acceptable for module-load-time gating but inside class methods prefer routing through the backend.
- `pyproject.toml` contains placeholder tokens (`#@NAMESUFFIX@`, `#@SKIP_PLUGIN@`, `#@DEPS_*@`, `#@FALLBACK_VERSION@`, `#@ADD_CUDA_HERE` in CMakeLists.txt) consumed by the plugin-wheel build pipeline (`gpubackendtools-cuda{11,12,13}x`). Don't remove them when editing those files.
- `dist/` contains pre-built wheels from prior local builds — not source of truth.
- `docs/doctrees/`, `src/gpubackendtools/_version.py` (auto-generated by `setuptools_scm`), and various untracked `.pyx` / `.pxd` are dev artifacts — leave them alone unless specifically asked.
- Build with `pip` / `cmake` only — there is no Makefile-based workflow.

## Backend implementation hierarchy (sprint-wide rule)

When implementing or modifying an algorithm that exists across multiple
backends (GPU C++ / CPU C++ / JAX), follow this hierarchy:

1. **GPU C++ (CUDA) leads.** This is the canonical performance target
   and reference implementation. New algorithms and optimizations are
   designed for the GPU first; CPU and JAX paths follow.

2. **CPU C++ mirrors GPU C++ as closely as possible.** Same kernel
   structure, same algorithm, same data flow — use `#ifdef __CUDACC__`
   or shared compile-time macros (`CUDA_SHARED`, `THREAD_START`,
   `BLOCK_INCR`, …) to bridge platform differences. The CPU path
   exists primarily for testing and CPU-only environments; it must
   not diverge in algorithm or output beyond floating-point order of
   operations.

3. **CPU C++ must reproduce the overall lisatools computation.**
   Against the lisatools reference (e.g. `FDSignal.transform`,
   `TDSignal.transform`, `XYZ2SensitivityMatrix`), match to machine
   precision (≤ 1e-15 mismatch) in direct modes; cache/approximation
   modes have documented per-feature error budgets.

4. **JAX may diverge internally** — design it to be JAX-efficient.
   JAX-CPU and JAX-GPU compilation targets may even differ. Use
   JAX-native idioms (`jax.lax.scan`, `jax.vmap`, static-shape
   `dynamic_slice` + masks, functional carries) rather than
   mechanically translating CUDA shared memory / register caches.

5. **JAX must match C++ inner-product outputs.** End-to-end
   likelihood quantities (`<d|h>`, `<h|h>`, swap_ll 5 terms) must
   match the C++ to floating-point precision (reldiff ≲ 1e-12) on
   representative test cases. Intermediate quantities (raw templates,
   per-chunk WDM coefficients) may differ at FP precision due to
   summation order — validate at the inner-product level.

**Workflow for a new feature.** GPU C++ → CPU C++ via `#ifdef` → JAX
with JAX-native idioms → cross-backend inner-product validation.


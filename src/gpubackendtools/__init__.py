"""GPU Backend Tools."""

# ruff: noqa: E402
try:
    from gpubackendtools._version import (  # pylint: disable=E0401,E0611
        __version__,
        __version_tuple__,
    )

except ModuleNotFoundError:
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

    try:
        __version__ = version(__name__)
        __version_tuple__ = tuple(__version__.split("."))
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "unknown"
        __version_tuple__ = (0, 0, 0, "unknown")
    finally:
        del version, PackageNotFoundError

_is_editable: bool
try:
    from . import _editable

    _is_editable = True
    del _editable
except (ModuleNotFoundError, ImportError):
    _is_editable = False


def get_include() -> str:
    """Absolute path to the directory containing GPUBackendTools' public C++/CUDA headers.

    Downstream packages (LISAanalysistools, GBGPU, BBHx, FastEMRIWaveforms)
    add this to their compiler include path so that ``#include "gbt_global.h"``,
    ``#include "cuda_complex.hpp"``, ``#include "InterpolateDevice.hh"``, and
    ``#include "pybind11_cuda_array_interface.hpp"`` resolve against the
    installed wheel's bundled headers (no sprint-tree path required).

    Example (CMake shell-out):

        execute_process(
          COMMAND ${Python_EXECUTABLE} -c
          "import gpubackendtools; print(gpubackendtools.get_include())"
          OUTPUT_VARIABLE GBT_INCLUDE_DIR
          OUTPUT_STRIP_TRAILING_WHITESPACE)
        target_include_directories(my_target PRIVATE ${GBT_INCLUDE_DIR})
    """
    import os.path
    return os.path.join(os.path.dirname(__file__), "cutils")


def get_cmake_module_path() -> str:
    """Absolute path to the directory containing ``GPUBackendToolsConfig.cmake``.

    For downstreams that prefer ``find_package(GPUBackendTools CONFIG)`` over
    a Python shell-out:

        execute_process(
          COMMAND ${Python_EXECUTABLE} -c
          "import gpubackendtools; print(gpubackendtools.get_cmake_module_path())"
          OUTPUT_VARIABLE GBT_CMAKE_DIR
          OUTPUT_STRIP_TRAILING_WHITESPACE)
        list(APPEND CMAKE_PREFIX_PATH ${GBT_CMAKE_DIR})
        find_package(GPUBackendTools CONFIG REQUIRED)

    Returns the same ``cutils/`` directory that ships ``cmake_functions.cmake``
    plus the installed ``GPUBackendToolsConfig.cmake``.
    """
    import os.path
    return os.path.join(os.path.dirname(__file__), "cutils")


from . import cutils, utils
from .globals import get_backend, get_first_backend, has_backend, Globals

from .pointeradjust import wrapper, pointer_adjust

from .parallelbase import ParallelModuleBase

from .globals import Globals
from .cutils import GBTCpuBackend, GBTCuda11xBackend, GBTCuda12xBackend, GBTCuda13xBackend

add_backends = {
    "gbt_cpu": GBTCpuBackend,
    "gbt_cuda11x": GBTCuda11xBackend,
    "gbt_cuda12x": GBTCuda12xBackend,
    "gbt_cuda13x": GBTCuda13xBackend,
}

# Pure-JAX backend (subpackage gated on `import jax`). Optional --
# users without JAX keep working with just the C++/CUDA backends.
try:
    from .jax import GBTJaxBackend as _GBTJaxBackend
    if _GBTJaxBackend is not None:
        add_backends["gbt_jax"] = _GBTJaxBackend
except (ImportError, ModuleNotFoundError):
    pass

Globals().backends_manager.add_backends(add_backends)


__all__ = [
    "__version__",
    "__version_tuple__",
    "_is_editable",
    # "amplitude",
    # "cutils",
    # "files",
    # "summation",
    # "trajectory",
    # "utils",
    # "waveform",
    "get_include",
    "get_cmake_module_path",
    "get_logger",
    "get_config",
    "get_config_setter",
    "get_backend",
    "get_first_backend",
    "get_file_manager",
    "has_backend",
    "ParallelModuleBase",
    "wrapper",
    "pointer_adjust",
    "Globals",
]

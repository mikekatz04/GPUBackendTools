from __future__ import annotations
import dataclasses
import enum
import types
import typing
import abc
from typing import Optional, Sequence, TypeVar, Union

from ..gpubackendtools import BackendMethods, CpuBackend, Cuda11xBackend, Cuda12xBackend
from ..exceptions import *

@dataclasses.dataclass
class GBTBackendMethods(BackendMethods):
    interpolate_wrap: typing.Callable[(...), None]
    pyCubicSplineWrap: object

class GBTBackend:
    interpolate_wrap: typing.Callable[(...), None]
    pyCubicSplineWrap: object

    def __init__(self, gbt_backend_methods):

        # set direct gbt methods
        # pass rest to general backend
        assert isinstance(gbt_backend_methods, GBTBackendMethods)

        self.interpolate_wrap = gbt_backend_methods.interpolate_wrap
        self.pyCubicSplineWrap = gbt_backend_methods.pyCubicSplineWrap


class GBTCpuBackend(CpuBackend, GBTBackend):
    """Implementation of the CPU backend"""
    
    _backend_name = "gbt_backend_cpu"
    _name = "gbt_cpu"
    def __init__(self, *args, **kwargs):
        CpuBackend.__init__(self, *args, **kwargs)
        GBTBackend.__init__(self, self.cpu_methods_loader())

    @staticmethod
    def cpu_methods_loader() -> GBTBackendMethods:
        try:
            import gbt_backend_cpu.interp
            
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cpu' backend could not be imported."
            ) from e

        numpy = GBTCpuBackend.check_numpy()

        return GBTBackendMethods(
            interpolate_wrap=gbt_backend_cpu.interp.interpolate_wrap,
            pyCubicSplineWrap=gbt_backend_cpu.interp.pyCubicSplineWrap,
            xp=numpy,
        )


class GBTCuda11xBackend(Cuda11xBackend, GBTBackend):

    """Implementation of CUDA 11.x backend"""
    _backend_name : str = "gbt_backend_cuda11x"
    _name = "gbt_cuda11x"

    def __init__(self, *args, **kwargs):
        Cuda11xBackend.__init__(self, *args, **kwargs)
        GBTBackend.__init__(self, self.cuda11x_module_loader())
        
    @staticmethod
    def cuda11x_module_loader():
        try:
            import gbt_backend_cuda11x.interp

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda11x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda11x' backend requires cupy", pip_deps=["cupy-cuda11x"]
            ) from e

        return GBTBackendMethods(
            interpolate_wrap=gbt_backend_cuda11x.interp.interpolate_wrap,
            pyCubicSplineWrap=gbt_backend_cuda11x.interp.pyCubicSplineWrap,
            xp=cupy,
        )

class GBTCuda12xBackend(Cuda12xBackend, GBTBackend):
    """Implementation of CUDA 12.x backend"""
    _backend_name : str = "gbt_backend_cuda12x"
    _name = "gbt_cuda12x"
    
    def __init__(self, *args, **kwargs):
        Cuda12xBackend.__init__(self, *args, **kwargs)
        GBTBackend.__init__(self, self.cuda12x_module_loader())
        
    @staticmethod
    def cuda12x_module_loader():
        try:
            import gbt_backend_cuda12x.interp

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda12x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda12x' backend requires cupy", pip_deps=["cupy-cuda12x"]
            ) from e

        return GBTBackendMethods(
            interpolate_wrap=gbt_backend_cuda12x.interp.interpolate_wrap,
            pyCubicSplineWrap=gbt_backend_cuda12x.interp.pyCubicSplineWrap,
            xp=cupy,
        )

"""List of existing backends, per default order of preference."""
# TODO: __all__ ?



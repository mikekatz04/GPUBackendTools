from __future__ import annotations
import dataclasses
import enum
import types
import typing
import abc
from typing import Optional, Sequence, TypeVar, Union

from ..gpubackendtools import BackendMethods, CpuBackend, Cuda11xBackend, Cuda12xBackend, Cuda13xBackend
from ..exceptions import *

@dataclasses.dataclass
class GBTBackendMethods(BackendMethods):
    interpolate_wrap: typing.Callable[(...), None]
    fit_cubic_spline_thomas: typing.Callable[(...), None]
    fit_cubic_spline_pcr: typing.Optional[typing.Callable[(...), None]]
    CubicSplineWrap: object
    CubicSpline: object
    # Quintic (k=5) spline methods. Optional/default-None so backends that do
    # not provide them (e.g. the pure-JAX backend) need not set them.
    interpolate_quintic_wrap: typing.Optional[typing.Callable[(...), None]] = None
    QuinticSplineWrap: typing.Optional[object] = None
    QuinticSpline: typing.Optional[object] = None

class GBTBackend:
    interpolate_wrap: typing.Callable[(...), None]
    fit_cubic_spline_thomas: typing.Callable[(...), None]
    fit_cubic_spline_pcr: typing.Optional[typing.Callable[(...), None]]
    CubicSplineWrap: object
    CubicSpline: object
    interpolate_quintic_wrap: typing.Optional[typing.Callable[(...), None]]
    QuinticSplineWrap: typing.Optional[object]
    QuinticSpline: typing.Optional[object]

    def __init__(self, gbt_backend_methods):

        # set direct gbt methods
        # pass rest to general backend
        assert isinstance(gbt_backend_methods, GBTBackendMethods)

        self.interpolate_wrap = gbt_backend_methods.interpolate_wrap
        self.fit_cubic_spline_thomas = gbt_backend_methods.fit_cubic_spline_thomas
        self.fit_cubic_spline_pcr = gbt_backend_methods.fit_cubic_spline_pcr
        self.CubicSplineWrap = gbt_backend_methods.CubicSplineWrap
        self.CubicSpline = gbt_backend_methods.CubicSpline
        self.interpolate_quintic_wrap = gbt_backend_methods.interpolate_quintic_wrap
        self.QuinticSplineWrap = gbt_backend_methods.QuinticSplineWrap
        self.QuinticSpline = gbt_backend_methods.QuinticSpline


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
            fit_cubic_spline_thomas=gbt_backend_cpu.interp.fit_cubic_spline_thomas,
            fit_cubic_spline_pcr=None,
            CubicSplineWrap=gbt_backend_cpu.interp.CubicSplineWrapCPU,
            CubicSpline=gbt_backend_cpu.interp.CubicSplineCPU,
            interpolate_quintic_wrap=gbt_backend_cpu.interp.interpolate_quintic_wrap,
            QuinticSplineWrap=gbt_backend_cpu.interp.QuinticSplineWrapCPU,
            QuinticSpline=gbt_backend_cpu.interp.QuinticSplineCPU,
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
            fit_cubic_spline_thomas=None,
            fit_cubic_spline_pcr=gbt_backend_cuda11x.interp.fit_cubic_spline_pcr,
            CubicSplineWrap=gbt_backend_cuda11x.interp.CubicSplineWrapGPU,
            CubicSpline=gbt_backend_cuda11x.interp.CubicSplineGPU,
            interpolate_quintic_wrap=gbt_backend_cuda11x.interp.interpolate_quintic_wrap,
            QuinticSplineWrap=gbt_backend_cuda11x.interp.QuinticSplineWrapGPU,
            QuinticSpline=gbt_backend_cuda11x.interp.QuinticSplineGPU,
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
            fit_cubic_spline_thomas=None,
            fit_cubic_spline_pcr=gbt_backend_cuda12x.interp.fit_cubic_spline_pcr,
            CubicSplineWrap=gbt_backend_cuda12x.interp.CubicSplineWrapGPU,
            CubicSpline=gbt_backend_cuda12x.interp.CubicSplineGPU,
            interpolate_quintic_wrap=gbt_backend_cuda12x.interp.interpolate_quintic_wrap,
            QuinticSplineWrap=gbt_backend_cuda12x.interp.QuinticSplineWrapGPU,
            QuinticSpline=gbt_backend_cuda12x.interp.QuinticSplineGPU,
            xp=cupy,
        )


class GBTCuda13xBackend(Cuda13xBackend, GBTBackend):
    """Implementation of CUDA 12.x backend"""
    _backend_name : str = "gbt_backend_cuda13x"
    _name = "gbt_cuda13x"
    
    def __init__(self, *args, **kwargs):
        Cuda13xBackend.__init__(self, *args, **kwargs)
        GBTBackend.__init__(self, self.cuda13x_module_loader())
        
    @staticmethod
    def cuda13x_module_loader():
        try:
            import gbt_backend_cuda13x.interp

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda13x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda13x' backend requires cupy", pip_deps=["cupy-cuda13x"]
            ) from e

        return GBTBackendMethods(
            interpolate_wrap=gbt_backend_cuda13x.interp.interpolate_wrap,
            fit_cubic_spline_thomas=None,
            fit_cubic_spline_pcr=gbt_backend_cuda13x.interp.fit_cubic_spline_pcr,
            CubicSplineWrap=gbt_backend_cuda13x.interp.CubicSplineWrapGPU,
            CubicSpline=gbt_backend_cuda13x.interp.CubicSplineGPU,
            interpolate_quintic_wrap=gbt_backend_cuda13x.interp.interpolate_quintic_wrap,
            QuinticSplineWrap=gbt_backend_cuda13x.interp.QuinticSplineWrapGPU,
            QuinticSpline=gbt_backend_cuda13x.interp.QuinticSplineGPU,
            xp=cupy,
        )

"""List of existing backends, per default order of preference."""
# TODO: __all__ ?



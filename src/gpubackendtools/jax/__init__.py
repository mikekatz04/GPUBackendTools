"""JAX backend for gpubackendtools.

Mirrors the CPU / CUDA backends in :mod:`gpubackendtools.cutils` but
uses ``jax.numpy`` and pure-Python (autograd-friendly) implementations
of the spline-fit / spline-eval kernels. Downstream packages
(``lisatools.jax``, ``fastlisaresponse.jax``, ...) consume this
backend the same way they consume the cuda/cpu ones -- via the
``backend.CubicSplineWrap`` / ``backend.interpolate_wrap`` slots on
the resolved :class:`GBTBackend` instance.

The subpackage import is gated on ``import jax`` so users without
JAX keep working with the C++/CUDA backends unmodified.
"""
from __future__ import annotations

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    _HAS_JAX = True
except (ImportError, ModuleNotFoundError):
    _HAS_JAX = False

if _HAS_JAX:
    from .backend import GBTJaxBackend
    from .cubic_spline import (
        CubicSplineWrapJAX,
        fit_cubic_spline_not_a_knot,
        evaluate_cubic_spline,
        interpolate_wrap_jax,
    )

    __all__ = [
        "GBTJaxBackend",
        "CubicSplineWrapJAX",
        "fit_cubic_spline_not_a_knot",
        "evaluate_cubic_spline",
        "interpolate_wrap_jax",
    ]
else:
    __all__ = []

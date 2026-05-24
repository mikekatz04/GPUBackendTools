"""JAX-native cubic-spline fit + evaluation.

This is the JAX analog of the C++ ``interpolate_wrap`` /
``CubicSplineWrap`` exposed by the ``gbt_backend_cpu`` /
``gbt_backend_cudaXXx`` plugin modules. Used by
:class:`gpubackendtools.interpolate.CubicSplineInterpolant` when the
selected backend is ``gbt_jax``.

**Fit**: ``scipy.interpolate.CubicSpline`` with ``bc_type='not-a-knot'``
(matches the C++ side). The fit runs once on the host at construction
time and the resulting coefficients are stashed as ``jnp`` arrays.
Autograd flows through the evaluation, not the fit -- which matches
the use case (orbits / templates fit once, evaluated many times with
gradients).

**Eval**: pure JAX. Vectorized over the binary axis via ``jax.vmap``;
``jax.jit`` + ``jax.grad`` work cleanly.

The per-segment cubic form
        y(x) = y_i + c1_i*(x-x_i) + c2_i*(x-x_i)^2 + c3_i*(x-x_i)^3
is the same one the C++ wrapper exposes.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


# --------------------------------------------------------------------
# Single-spline fit (scipy under the hood). Host-side, NOT traced.
# --------------------------------------------------------------------
def _fit_one_not_a_knot_scipy(x_np: np.ndarray, y_np: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one not-a-knot cubic spline using scipy.

    Returns ``(c1, c2, c3)`` arrays of length ``N``. The last entry of
    each is unused at evaluation time but kept for layout parity with
    the C++ ``(ninterps*length,)`` storage.
    """
    from scipy.interpolate import CubicSpline as _SciCubicSpline
    sp = _SciCubicSpline(x_np, y_np, bc_type="not-a-knot")
    # scipy stores polynomial in (x - x_i) basis as:
    #   sp.c[3, i] = y_i  (= constant term)
    #   sp.c[2, i] = c1_i (first derivative)
    #   sp.c[1, i] = c2_i (1/2 second derivative)
    #   sp.c[0, i] = c3_i (1/6 third derivative)
    # shapes: sp.c has shape (4, N-1)
    c1_seg = sp.c[2, :]
    c2_seg = sp.c[1, :]
    c3_seg = sp.c[0, :]
    # Pad the last entry with 0 so storage matches y length N.
    c1 = np.concatenate([c1_seg, [0.0]])
    c2 = np.concatenate([c2_seg, [0.0]])
    c3 = np.concatenate([c3_seg, [0.0]])
    return c1, c2, c3


# --------------------------------------------------------------------
# Batched fit over ``ninterps`` independent splines.
# --------------------------------------------------------------------
def fit_cubic_spline_not_a_knot(x_flat, y_flat, length: int, ninterps: int
                                ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fit ``ninterps`` independent splines stacked in flat arrays.

    Inputs are ``(ninterps * length,)``; outputs are the same. Done
    host-side via scipy then transferred to JAX device.
    """
    x_np = np.asarray(x_flat).reshape(ninterps, length)
    y_np = np.asarray(y_flat).reshape(ninterps, length)
    c1 = np.empty_like(y_np)
    c2 = np.empty_like(y_np)
    c3 = np.empty_like(y_np)
    for i in range(ninterps):
        c1[i], c2[i], c3[i] = _fit_one_not_a_knot_scipy(x_np[i], y_np[i])
    return (
        jnp.asarray(c1.reshape(-1)),
        jnp.asarray(c2.reshape(-1)),
        jnp.asarray(c3.reshape(-1)),
    )


def interpolate_wrap_jax(x_flat, y_flat, B_flat, c1_flat, c2_flat, c3_flat,
                         length, ninterps):
    """JAX analog of ``interpolate_wrap``.

    The C++ signature mutates ``c1_flat`` / ``c2_flat`` / ``c3_flat``
    in place. JAX arrays are immutable, so this function instead
    returns the fitted ``(c1, c2, c3)`` and the caller is expected to
    rebind them. :class:`gpubackendtools.interpolate.CubicSplineInterpolant`
    branches on the JAX backend to use the returned values rather
    than re-using the input buffers.
    """
    return fit_cubic_spline_not_a_knot(x_flat, y_flat, length, ninterps)


# --------------------------------------------------------------------
# Spline evaluation (pure JAX -- autograd-compatible).
# --------------------------------------------------------------------
def evaluate_cubic_spline(x: jnp.ndarray, y: jnp.ndarray,
                          c1: jnp.ndarray, c2: jnp.ndarray, c3: jnp.ndarray,
                          x_new: jnp.ndarray, derivative: int = 0
                          ) -> jnp.ndarray:
    """Evaluate one cubic spline at ``x_new``.

    Args:
        x, y, c1, c2, c3: 1D arrays of length ``N``; the spline data
            from :func:`fit_cubic_spline_not_a_knot`.
        x_new: query points (any shape).
        derivative: 0..3; matches
            :meth:`gpubackendtools.interpolate.CubicSplineInterpolant.__call__`.

    Out-of-range ``x_new`` is clamped to the spline domain. Callers
    that want NaN / zero on out-of-bounds should mask the output.
    """
    N = x.shape[0]
    flat = x_new.reshape(-1)
    seg = jnp.clip(jnp.searchsorted(x, flat, side="right") - 1, 0, N - 2)
    x0 = x[seg]
    y0 = y[seg]
    c1s = c1[seg]
    c2s = c2[seg]
    c3s = c3[seg]
    dx = flat - x0
    if derivative == 0:
        out = y0 + c1s * dx + c2s * dx**2 + c3s * dx**3
    elif derivative == 1:
        out = c1s + 2.0 * c2s * dx + 3.0 * c3s * dx**2
    elif derivative == 2:
        out = 2.0 * c2s + 6.0 * c3s * dx
    elif derivative == 3:
        out = 6.0 * c3s
    else:
        raise ValueError("derivative must be 0..3")
    return out.reshape(x_new.shape)


# --------------------------------------------------------------------
# CubicSplineWrapJAX -- mirrors the C++ ``CubicSplineWrap`` API.
# --------------------------------------------------------------------
class CubicSplineWrapJAX:
    """Pure-Python analog of ``CubicSplineWrap`` / ``CubicSpline``.

    Constructor signature matches the C++ wrapper exactly so the
    existing call site
        backend.CubicSplineWrap(x_flat, y_flat, c1_flat, c2_flat,
                                c3_flat, ninterps, length, spline_type)
    works unchanged on the JAX backend.
    """

    def __init__(self, x_flat, y_flat, c1_flat, c2_flat, c3_flat,
                 ninterps: int, length: int, spline_type: int):
        self.x_flat = jnp.asarray(x_flat)
        self.y_flat = jnp.asarray(y_flat)
        self.c1_flat = jnp.asarray(c1_flat)
        self.c2_flat = jnp.asarray(c2_flat)
        self.c3_flat = jnp.asarray(c3_flat)
        self.ninterps = int(ninterps)
        self.length = int(length)
        self.spline_type = int(spline_type)

    def evaluate_one(self, interp_index: int, x_new, derivative: int = 0):
        """Evaluate spline ``interp_index`` at ``x_new``.

        Convenience for downstream consumers that don't want to manage
        the ``(ninterps*length,)`` layout by hand.
        """
        start = int(interp_index) * self.length
        stop = start + self.length
        return evaluate_cubic_spline(
            self.x_flat[start:stop],
            self.y_flat[start:stop],
            self.c1_flat[start:stop],
            self.c2_flat[start:stop],
            self.c3_flat[start:stop],
            x_new, derivative=derivative,
        )

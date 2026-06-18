"""Tests for the chunked / SPIKE-style quintic banded solve.

The `_chunk` hook on QuinticSplineInterpolant lets these tests force the
multi-chunk + reduced-system + recursion paths on small problem sizes by
passing a tiny chunk (e.g. 8). Correctness oracle is
scipy.interpolate.make_interp_spline(x, y, k=5). See
docs/superpowers/plans/2026-06-18-quintic-gpu-spike-solve.md.
"""
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
        else:  # random (strictly increasing)
            xrow = np.sort(rng.uniform(0.0, 1.0, n))
            xrow[0], xrow[-1] = 0.0, 1.0
        x = np.broadcast_to(xrow, (ninterps, n)).copy()
        y = (np.sin(6 * x) + 0.1 * rng.standard_normal((ninterps, n))).copy()
        x_new = np.linspace(xrow[0], xrow[-1], 97)
        spl = QuinticSplineInterpolant(x, y, force_backend="cpu", _chunk=chunk)
        got = spl(np.broadcast_to(x_new, (ninterps, 97)).copy())
        for i in range(ninterps):
            exp = _scipy_eval(xrow, y[i], x_new)
            np.testing.assert_allclose(got[i], exp, atol=tol, rtol=0)

    # Task 1: plumbing + single-chunk (chunk auto) still matches scipy.
    def test_single_chunk_uniform(self):
        self._check(n=40, ninterps=1, kind="uniform", chunk=0)


if __name__ == "__main__":
    unittest.main()

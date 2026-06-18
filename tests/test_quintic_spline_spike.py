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
    def _check(self, n, ninterps=1, kind="uniform", chunk=0, rtol=1e-10, atol=1e-12):
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
            # magnitude-aware: the quintic can oscillate to O(10) on noisy random
            # grids, so a pure absolute tol is wrong. SPIKE is exact (rel ~1e-13).
            np.testing.assert_allclose(got[i], exp, rtol=rtol, atol=atol)

    # Task 1: plumbing + single-chunk (chunk auto) still matches scipy.
    def test_single_chunk_uniform(self):
        self._check(n=40, ninterps=1, kind="uniform", chunk=0)

    # Task 2: tiny chunk forces the multi-chunk + reduced-system path.
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

    # Task 3: chunk large enough that the reduced order at least halves -> the
    # reduced solve recurses (R > C and 2R <= n). chunk=16/20, n=400/800.
    def test_two_level_reduced_uniform(self):
        self._check(n=400, ninterps=1, kind="uniform", chunk=16)
        self._check(n=800, ninterps=1, kind="uniform", chunk=20)

    def test_two_level_reduced_nonuniform(self):
        self._check(n=400, ninterps=2, kind="random", chunk=20)

    # Task 4: uniform grid exercises the LINEAR_SPACING Toeplitz cache path;
    # result must still match scipy (cache is correctness-preserving).
    def test_uniform_fastpath_matches_scipy(self):
        self._check(n=300, ninterps=3, kind="uniform", chunk=16)
        self._check(n=512, ninterps=1, kind="uniform", chunk=32)


if __name__ == "__main__":
    unittest.main()

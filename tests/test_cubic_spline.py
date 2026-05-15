import unittest
import numpy as np
import warnings
import os

path_to_file = os.path.dirname(__file__)

from gpubackendtools.interpolate import CubicSplineInterpolant

try:
    import cupy as cp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    pass

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False

from scipy.interpolate import CubicSpline as CubicSpline_scipy

CUBIC_SPLINE_LINEAR_SPACING = 1
CUBIC_SPLINE_LOG10_SPACING = 2

class CubicSplineTest(unittest.TestCase):
    def test_cubic_spline(self):
        force_backend = "cpu" if not gpu_available else "gpu"
        
        N = 1000
        x_in = np.linspace(0.0, 1.0, N)
        y_in = x_in ** 3 + x_in ** 2 + x_in ** 1 + x_in
        
        spl_scipy = CubicSpline_scipy(x_in, y_in)

        our_spl = CubicSplineInterpolant(x_in[None, :], y_in[None, :], force_backend="cpu")
        x_new = np.random.uniform(x_in[0], x_in[-1], size=10000)
        scipy_check = spl_scipy(x_new)

        our_check = our_spl(x_new[None, :])
        self.assertTrue(np.allclose(our_check[0], scipy_check))

    def test_cubic_spline_tiled(self):
        force_backend = "cpu" if not gpu_available else "gpu"
        
        N = 1000
        _x = np.linspace(0.0, 1.0, N)
        x_in = np.tile(_x, (2, 2, 1))
        y_in = np.tile((_x ** 3 + _x ** 2 + _x ** 1 + _x), (2, 2, 1))
        
        spl_scipy = CubicSpline_scipy(x_in[0, 0], y_in[0, 0])

        our_spl = CubicSplineInterpolant(x_in, y_in, force_backend="cpu")

        _x_new = np.random.uniform(_x[0], _x[-1], size=10000)
        scipy_check = spl_scipy(_x_new)
        x_new = np.tile(_x_new, (2, 2, 1))

        our_check = our_spl(x_new)
        self.assertTrue(np.allclose(our_check[0, 0], scipy_check))

    def test_cubic_spline_indexed(self):
        force_backend = "cpu" if not gpu_available else "gpu"
        
        N = 1000
        _x = np.linspace(0.0, 1.0, N)
        x_in = np.tile(_x, (2, 2, 1))
        y_in = np.tile((_x ** 3 + _x ** 2 + _x ** 1 + _x), (2, 2, 1))
        
        spl_scipy = CubicSpline_scipy(x_in[0, 0], y_in[0, 0])

        our_spl = CubicSplineInterpolant(x_in, y_in, force_backend="cpu")

        _x_new = np.random.uniform(_x[0], _x[-1], size=10000)
        scipy_check = spl_scipy(_x_new)
        x_new = np.tile(_x_new, (2, 1))
        ind_interps = np.array([1, 3])

        our_check = our_spl(x_new, ind_interps=ind_interps)
        self.assertTrue(np.allclose(our_check[0], scipy_check))

    def test_cubic_spline_c_backend(self):
        force_backend = "cpu" if not gpu_available else "gpu"
        xp = cp if gpu_available else np
        N = 1000
        _x = np.linspace(0.0, 1.0, N)
        x_in = np.tile(_x, (2, 2, 1))
        y_in = np.tile((_x ** 3 + _x ** 2 + _x ** 1 + _x), (2, 2, 1))

        spl_scipy = CubicSpline_scipy(x_in[0, 0], y_in[0, 0])

        our_spl = CubicSplineInterpolant(xp.asarray(x_in), xp.asarray(y_in), force_backend=force_backend)

        cpp_class = our_spl.cpp_class
        _x_new = np.random.uniform(_x[0], _x[-1], size=10000)
        scipy_check = spl_scipy(_x_new)\

        _y_new = xp.zeros_like(_x_new)
        cpp_class.eval_wrap(_y_new, xp.asarray(_x_new), xp.zeros_like(_x_new, dtype=np.int32), len(_x_new))
        if gpu_available:
            _y_new = _y_new.get()
        self.assertTrue(np.allclose(_y_new, scipy_check))


class SingleSplineFitTest(unittest.TestCase):
    """Test the per-spline Thomas and PCR device functions exposed via pybind11.

    Each fit function fills (c1, c2, c3) in place from (x, y); we evaluate the
    resulting piecewise cubic at a grid of new points and compare against
    scipy's not-a-knot CubicSpline.
    """

    @staticmethod
    def _evaluate(x_new, x, y, c1, c2, c3, xp):
        inds = xp.searchsorted(x, x_new, side="right") - 1
        inds = xp.clip(inds, 0, len(x) - 2)
        dx = x_new - x[inds]
        return y[inds] + c1[inds] * dx + c2[inds] * dx ** 2 + c3[inds] * dx ** 3

    def _run_fit_and_compare(self, fit_func, xp, atol=1e-9):
        N = 128
        x_np = np.linspace(0.0, 1.0, N)
        y_np = x_np ** 3 + x_np ** 2 + x_np + 1.0

        x = xp.asarray(x_np)
        y = xp.asarray(y_np)
        c1 = xp.zeros(N, dtype=np.float64)
        c2 = xp.zeros(N, dtype=np.float64)
        c3 = xp.zeros(N, dtype=np.float64)
        B = xp.zeros(N, dtype=np.float64)

        fit_func(x, y, c1, c2, c3, B, N, CUBIC_SPLINE_LINEAR_SPACING)

        x_new_np = np.linspace(x_np[0] + 1e-6, x_np[-1] - 1e-6, 5000)
        scipy_y = CubicSpline_scipy(x_np, y_np)(x_new_np)

        x_new = xp.asarray(x_new_np)
        our_y = self._evaluate(x_new, x, y, c1, c2, c3, xp)
        if hasattr(our_y, "get"):
            our_y = our_y.get()

        self.assertTrue(np.allclose(our_y, scipy_y, atol=atol))

    def test_thomas_cpu(self):
        from gpubackendtools import get_backend
        backend = get_backend("gbt_cpu")
        self._run_fit_and_compare(backend.fit_cubic_spline_thomas, np)

    @unittest.skipUnless(gpu_available, "GPU not available")
    def test_pcr(self):
        from gpubackendtools import get_backend
        backend = get_backend("gbt_gpu")
        self._run_fit_and_compare(backend.fit_cubic_spline_pcr, cp)

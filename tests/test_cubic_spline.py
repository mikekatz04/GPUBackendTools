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
        
        N = 1000
        _x = np.linspace(0.0, 1.0, N)
        x_in = np.tile(_x, (2, 2, 1))
        y_in = np.tile((_x ** 3 + _x ** 2 + _x ** 1 + _x), (2, 2, 1))
        
        spl_scipy = CubicSpline_scipy(x_in[0, 0], y_in[0, 0])

        our_spl = CubicSplineInterpolant(x_in, y_in, force_backend="cpu")

        cpp_class = our_spl.cpp_class
        _x_new = np.random.uniform(_x[0], _x[-1], size=10000)
        scipy_check = spl_scipy(_x_new)

        _y_new = np.zeros_like(_x_new)
        cpp_class.eval(_y_new, _x_new, np.zeros_like(_x_new, dtype=np.int32), len(_x_new))
        self.assertTrue(np.allclose(_y_new, scipy_check))

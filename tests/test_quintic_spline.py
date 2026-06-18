import unittest
import numpy as np
import warnings
import os

path_to_file = os.path.dirname(__file__)

from gpubackendtools.interpolate import CubicSplineInterpolant, QuinticSplineInterpolant

try:
    import cupy as cp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    pass

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False

from scipy.interpolate import make_interp_spline

CUBIC_SPLINE_LINEAR_SPACING = 1
CUBIC_SPLINE_LOG10_SPACING = 2


def _scipy_quintic(x, y):
    """scipy not-a-knot quintic (bc_type=None) -- the reference oracle."""
    return make_interp_spline(x, y, k=5)


class QuinticSplineTest(unittest.TestCase):
    """QuinticSplineInterpolant vs scipy.interpolate.make_interp_spline(k=5)."""

    def test_quintic_spline(self):
        N = 1000
        x_in = np.linspace(0.0, 1.0, N)
        y_in = np.sin(6.0 * x_in) + x_in ** 2

        spl_scipy = _scipy_quintic(x_in, y_in)

        our_spl = QuinticSplineInterpolant(x_in[None, :], y_in[None, :], force_backend="cpu")
        x_new = np.random.uniform(x_in[0], x_in[-1], size=10000)
        scipy_check = spl_scipy(x_new)

        our_check = our_spl(x_new[None, :])
        self.assertTrue(np.allclose(our_check[0], scipy_check, atol=1e-9))

    def test_quintic_spline_quintic_poly_exact(self):
        # A degree-5 polynomial must be reproduced (essentially) exactly.
        N = 200
        x_in = np.linspace(-1.0, 2.0, N)
        y_in = 1 + 2 * x_in - 3 * x_in ** 2 + 0.5 * x_in ** 3 + x_in ** 4 - 2 * x_in ** 5

        our_spl = QuinticSplineInterpolant(x_in[None, :], y_in[None, :], force_backend="cpu")
        x_new = np.linspace(x_in[0], x_in[-1], 5000)
        truth = 1 + 2 * x_new - 3 * x_new ** 2 + 0.5 * x_new ** 3 + x_new ** 4 - 2 * x_new ** 5

        our_check = our_spl(x_new[None, :])
        self.assertTrue(np.allclose(our_check[0], truth, atol=1e-7))

    def test_quintic_spline_tiled(self):
        N = 1000
        _x = np.linspace(0.0, 1.0, N)
        x_in = np.tile(_x, (2, 2, 1))
        y_in = np.tile(np.sin(5.0 * _x), (2, 2, 1))

        spl_scipy = _scipy_quintic(x_in[0, 0], y_in[0, 0])

        our_spl = QuinticSplineInterpolant(x_in, y_in, force_backend="cpu")

        _x_new = np.random.uniform(_x[0], _x[-1], size=10000)
        scipy_check = spl_scipy(_x_new)
        x_new = np.tile(_x_new, (2, 2, 1))

        our_check = our_spl(x_new)
        self.assertTrue(np.allclose(our_check[0, 0], scipy_check, atol=1e-9))

    def test_quintic_spline_indexed(self):
        N = 1000
        _x = np.linspace(0.0, 1.0, N)
        x_in = np.tile(_x, (2, 2, 1))
        y_in = np.tile(np.sin(5.0 * _x), (2, 2, 1))

        spl_scipy = _scipy_quintic(x_in[0, 0], y_in[0, 0])

        our_spl = QuinticSplineInterpolant(x_in, y_in, force_backend="cpu")

        _x_new = np.random.uniform(_x[0], _x[-1], size=10000)
        scipy_check = spl_scipy(_x_new)
        x_new = np.tile(_x_new, (2, 1))
        ind_interps = np.array([1, 3])

        our_check = our_spl(x_new, ind_interps=ind_interps)
        self.assertTrue(np.allclose(our_check[0], scipy_check, atol=1e-9))

    def test_quintic_spline_c_backend(self):
        force_backend = "cpu" if not gpu_available else "gpu"
        xp = cp if gpu_available else np
        N = 1000
        _x = np.linspace(0.0, 1.0, N)
        x_in = np.tile(_x, (2, 2, 1))
        y_in = np.tile(np.sin(5.0 * _x), (2, 2, 1))

        spl_scipy = _scipy_quintic(x_in[0, 0], y_in[0, 0])

        our_spl = QuinticSplineInterpolant(xp.asarray(x_in), xp.asarray(y_in), force_backend=force_backend)

        cpp_class = our_spl.cpp_class
        _x_new = np.random.uniform(_x[0], _x[-1], size=10000)
        scipy_check = spl_scipy(_x_new)

        _y_new = xp.zeros_like(_x_new)
        cpp_class.eval_wrap(_y_new, xp.asarray(_x_new), xp.zeros_like(_x_new, dtype=np.int32), len(_x_new))
        if gpu_available:
            _y_new = _y_new.get()
        self.assertTrue(np.allclose(_y_new, scipy_check, atol=1e-9))

    def test_quintic_log_spacing(self):
        # Fit is in real-x space; log spacing only affects eval-time window pick.
        N = 300
        x_in = np.logspace(-2.0, 1.0, N)
        y_in = 1.0 / x_in

        spl_scipy = _scipy_quintic(x_in, y_in)

        our_spl = QuinticSplineInterpolant(x_in[None, :], y_in[None, :], force_backend="cpu")
        self.assertEqual(our_spl.spline_type, CUBIC_SPLINE_LOG10_SPACING)

        x_new = np.random.uniform(x_in[0], x_in[-1], size=5000)
        self.assertTrue(np.allclose(our_spl(x_new[None, :])[0], spl_scipy(x_new), atol=1e-8))

    def test_quintic_derivatives(self):
        N = 200
        x_in = np.linspace(0.0, 1.0, N)
        y_in = np.sin(6.0 * x_in)

        spl_scipy = _scipy_quintic(x_in, y_in)
        our_spl = QuinticSplineInterpolant(x_in[None, :], y_in[None, :], force_backend="cpu")

        x_new = np.linspace(x_in[1], x_in[-2], 3000)
        # value + low-order derivatives are at/near machine precision; high orders
        # are conditioning-limited (s^(nu) ~ (1/dx)^nu), so the tolerance loosens.
        tols = {0: 1e-9, 1: 1e-7, 2: 1e-5, 3: 1e-3}
        for nu, atol in tols.items():
            ours = our_spl(x_new[None, :], derivative=nu)[0]
            ref = spl_scipy(x_new, nu=nu)
            self.assertTrue(
                np.allclose(ours, ref, atol=atol, rtol=1e-6),
                msg=f"derivative {nu} mismatch (max abs {np.max(np.abs(ours - ref)):.2e})",
            )

    def test_length_too_short_raises(self):
        x_in = np.linspace(0.0, 1.0, 5)
        y_in = x_in ** 2
        with self.assertRaises(ValueError):
            QuinticSplineInterpolant(x_in[None, :], y_in[None, :], force_backend="cpu")


class QuinticVsCubicCrossCheckTest(unittest.TestCase):
    """On degree<=3 data, the gbt cubic and gbt quintic both reproduce the
    polynomial exactly, so they must agree (and both agree with scipy k=5)."""

    def test_cubic_data_matches_gbt_cubic_and_scipy(self):
        N = 500
        x_in = np.linspace(0.0, 1.0, N)
        y_in = x_in ** 3 + x_in ** 2 + x_in

        cub = CubicSplineInterpolant(x_in[None, :], y_in[None, :], force_backend="cpu")
        qui = QuinticSplineInterpolant(x_in[None, :], y_in[None, :], force_backend="cpu")
        scipy5 = make_interp_spline(x_in, y_in, k=5)

        x_new = np.random.uniform(x_in[0], x_in[-1], size=8000)

        qui_y = qui(x_new[None, :])[0]
        cub_y = cub(x_new[None, :])[0]

        # quintic == gbt cubic (both reproduce the cubic exactly)
        self.assertTrue(np.allclose(qui_y, cub_y, atol=1e-9))
        # quintic == scipy make_interp_spline(k=5)
        self.assertTrue(np.allclose(qui_y, scipy5(x_new), atol=1e-9))
        # all three == the underlying polynomial
        truth = x_new ** 3 + x_new ** 2 + x_new
        self.assertTrue(np.allclose(qui_y, truth, atol=1e-9))


class QuinticSingleSplineFitTest(unittest.TestCase):
    """Exercise the native batched fit at ninterps=1 across small/large n,
    comparing the evaluated piecewise quintic against scipy k=5."""

    @staticmethod
    def _evaluate(x_new, x, y, c1, c2, c3, c4, c5, xp):
        inds = xp.searchsorted(x, x_new, side="right") - 1
        inds = xp.clip(inds, 0, len(x) - 2)
        dx = x_new - x[inds]
        return y[inds] + dx * (c1[inds] + dx * (c2[inds] + dx * (c3[inds]
                      + dx * (c4[inds] + dx * c5[inds]))))

    def _run_for_backend(self, backend_name, xp):
        from gpubackendtools import get_backend
        backend = get_backend(backend_name)

        for N in (6, 7, 128, 1000):
            x_np = np.linspace(0.0, 1.0, N)
            y_np = np.sin(5.0 * x_np) + x_np ** 2

            x = xp.asarray(x_np)
            y = xp.asarray(y_np)
            c1 = xp.zeros(N, dtype=np.float64)
            c2 = xp.zeros(N, dtype=np.float64)
            c3 = xp.zeros(N, dtype=np.float64)
            c4 = xp.zeros(N, dtype=np.float64)
            c5 = xp.zeros(N, dtype=np.float64)

            # ninterps = 1 single-spline fit through the batched native entry.
            backend.interpolate_quintic_wrap(x, y, c1, c2, c3, c4, c5, N, 1)

            x_new_np = np.linspace(x_np[0] + 1e-6, x_np[-1] - 1e-6, 5000)
            scipy_y = make_interp_spline(x_np, y_np, k=5)(x_new_np)

            x_new = xp.asarray(x_new_np)
            our_y = self._evaluate(x_new, x, y, c1, c2, c3, c4, c5, xp)
            if hasattr(our_y, "get"):
                our_y = our_y.get()

            self.assertTrue(
                np.allclose(our_y, scipy_y, atol=1e-9),
                msg=f"N={N} backend={backend_name} max abs {np.max(np.abs(our_y - scipy_y)):.2e}",
            )

    def test_cpu(self):
        self._run_for_backend("gbt_cpu", np)

    @unittest.skipUnless(gpu_available, "GPU not available")
    def test_gpu(self):
        self._run_for_backend("gbt_gpu", cp)


if __name__ == "__main__":
    unittest.main()

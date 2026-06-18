"""Visual comparison plots for the GBT spline interpolants.

Generates PNG figures comparing :class:`CubicSplineInterpolant` and
:class:`QuinticSplineInterpolant` against their SciPy reference oracles
(``make_interp_spline(k=3)`` / ``make_interp_spline(k=5)``):

1. ``overlay_behavior.png`` -- sparse-knot overlays on three representative
   functions, showing how cubic vs. quintic behave *between* knots and that
   each GBT spline lies on top of its SciPy counterpart.
2. ``agreement_residuals.png`` -- residuals (GBT - SciPy) on a dense grid for a
   smooth, densely-sampled function, confirming machine-precision (~1e-13)
   agreement.
3. ``derivatives.png`` -- 1st and 2nd derivatives of the GBT quintic vs. the
   SciPy k=5 derivatives, demonstrating derivative continuity/agreement.

Runnable as a script::

    uv run --no-sync python tests/plot_splines.py

and importable -- ``generate_all_plots()`` returns the list of written PNG
paths. Safe to run repeatedly and headless (uses the Agg backend, no
``plt.show()``). A small unittest at the bottom asserts the PNGs get created.
"""

import os
import unittest

import numpy as np

import matplotlib

# Non-interactive backend MUST be selected before importing pyplot so this
# works headless / in CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gpubackendtools.interpolate import (  # noqa: E402
    CubicSplineInterpolant,
    QuinticSplineInterpolant,
)
from scipy.interpolate import make_interp_spline  # noqa: E402


# Directory for the generated figures (created on demand).
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

# All GBT interpolants are built CPU-only in this environment.
_FORCE_BACKEND = "cpu"


def _gbt_cubic(x, y):
    """GBT cubic spline over 1-D ``x`` / ``y``."""
    return CubicSplineInterpolant(x[None, :], y[None, :], force_backend=_FORCE_BACKEND)


def _gbt_quintic(x, y):
    """GBT quintic spline over 1-D ``x`` / ``y``."""
    return QuinticSplineInterpolant(x[None, :], y[None, :], force_backend=_FORCE_BACKEND)


def _eval(spl, x_new, derivative=0):
    """Evaluate a GBT interpolant on a 1-D grid, returning a 1-D array."""
    return spl(x_new[None, :], derivative=derivative)[0]


def plot_overlay_behavior(path):
    """Sparse-knot overlays of cubic vs. quintic (GBT and SciPy).

    Three representative functions, each sampled at a deliberately *sparse* set
    of knots so the inter-knot behavior is visible. For each: the data knots are
    scattered, then dense curves of gbt-cubic, gbt-quintic, scipy-k3 and
    scipy-k5 are overlaid. The generating analytic function is drawn as a black
    dashed line beneath the interpolants, so their inter-knot deviation from the
    truth is visible.
    """
    cases = [
        (
            r"$\sin(2\pi x)$",
            lambda x: np.sin(2.0 * np.pi * x),
            np.linspace(0.0, 1.0, 12),
        ),
        (
            r"Runge $1/(1+25x^2)$",
            lambda x: 1.0 / (1.0 + 25.0 * x**2),
            np.linspace(-1.0, 1.0, 15),
        ),
        (
            r"$e^{-x}\sin(5x)$",
            lambda x: np.exp(-x) * np.sin(5.0 * x),
            np.linspace(0.0, 3.0, 10),
        ),
    ]

    fig, axes = plt.subplots(1, len(cases), figsize=(6.0 * len(cases), 5.0))
    if len(cases) == 1:
        axes = [axes]

    for ax, (title, func, x_knots) in zip(axes, cases):
        y_knots = func(x_knots)

        # Dense evaluation grid strictly inside the knot span.
        x_dense = np.linspace(x_knots[0], x_knots[-1], 1500)

        gbt_c = _eval(_gbt_cubic(x_knots, y_knots), x_dense)
        gbt_q = _eval(_gbt_quintic(x_knots, y_knots), x_dense)
        sc_c = make_interp_spline(x_knots, y_knots, k=3)(x_dense)
        sc_q = make_interp_spline(x_knots, y_knots, k=5)(x_dense)

        # Generating analytic curve (ground truth), drawn beneath the
        # interpolants so their inter-knot deviation from the truth is visible.
        ax.plot(x_dense, func(x_dense), color="k", ls="--", lw=1.0, zorder=1,
                label="analytic (truth)")
        ax.plot(x_dense, sc_c, color="tab:blue", lw=3.0, alpha=0.35,
                label="scipy k=3")
        ax.plot(x_dense, gbt_c, color="tab:blue", lw=1.2, ls="--",
                label="gbt cubic")
        ax.plot(x_dense, sc_q, color="tab:red", lw=3.0, alpha=0.35,
                label="scipy k=5")
        ax.plot(x_dense, gbt_q, color="tab:red", lw=1.2, ls="--",
                label="gbt quintic")
        ax.scatter(x_knots, y_knots, color="k", zorder=5, s=30,
                   label="knots")

        ax.set_title(f"{title}  ({x_knots.size} knots)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Cubic vs. Quintic interpolation behavior (GBT overlays SciPy)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_agreement_residuals(path):
    """Residuals GBT - SciPy on a dense grid (machine-precision check).

    A smooth function sampled at many knots; the residuals of gbt-quintic vs.
    scipy-k5 and gbt-cubic vs. scipy-k3 are plotted on a symlog y-axis to make
    the ~1e-13 floor obvious.
    """
    x_knots = np.linspace(0.0, 2.0 * np.pi, 80)
    y_knots = np.sin(x_knots) * np.exp(-0.1 * x_knots)

    x_dense = np.linspace(x_knots[0], x_knots[-1], 4000)

    gbt_c = _eval(_gbt_cubic(x_knots, y_knots), x_dense)
    gbt_q = _eval(_gbt_quintic(x_knots, y_knots), x_dense)
    sc_c = make_interp_spline(x_knots, y_knots, k=3)(x_dense)
    sc_q = make_interp_spline(x_knots, y_knots, k=5)(x_dense)

    res_c = gbt_c - sc_c
    res_q = gbt_q - sc_q

    max_c = np.max(np.abs(res_c))
    max_q = np.max(np.abs(res_q))

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax.plot(x_dense, res_c, color="tab:blue", lw=0.8,
            label=f"gbt cubic - scipy k=3  (max |.| = {max_c:.1e})")
    ax.plot(x_dense, res_q, color="tab:red", lw=0.8,
            label=f"gbt quintic - scipy k=5  (max |.| = {max_q:.1e})")

    # symlog keeps zero on-axis while spanning the tiny dynamic range.
    ax.set_yscale("symlog", linthresh=1e-15)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_title(
        "GBT vs. SciPy residuals on a dense grid "
        "(agreement at ~1e-13, machine precision)"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("residual (GBT - SciPy)")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_derivatives(path):
    """1st and 2nd derivatives of the GBT quintic vs. SciPy k=5 derivatives."""
    x_knots = np.linspace(0.0, 1.0, 40)
    y_knots = np.sin(6.0 * x_knots)

    q = _gbt_quintic(x_knots, y_knots)
    sc = make_interp_spline(x_knots, y_knots, k=5)

    # Stay strictly inside the knot span for a fair derivative comparison.
    x_dense = np.linspace(x_knots[1], x_knots[-2], 2000)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for ax, nu, label in zip(axes, (1, 2), ("1st", "2nd")):
        gbt_d = _eval(q, x_dense, derivative=nu)
        sc_d = sc.derivative(nu)(x_dense)
        max_d = np.max(np.abs(gbt_d - sc_d))

        ax.plot(x_dense, sc_d, color="tab:green", lw=3.0, alpha=0.35,
                label=f"scipy k=5 d{nu}")
        ax.plot(x_dense, gbt_d, color="tab:purple", lw=1.2, ls="--",
                label=f"gbt quintic d{nu}")
        ax.set_title(f"{label} derivative  (max |gbt - scipy| = {max_d:.1e})")
        ax.set_xlabel("x")
        ax.set_ylabel(f"d{nu}y / dx{nu}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        r"GBT quintic derivatives vs. SciPy k=5 ($y = \sin(6x)$)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def generate_all_plots(out_dir=PLOTS_DIR, verbose=True):
    """Generate every figure into ``out_dir``; return the list of PNG paths."""
    os.makedirs(out_dir, exist_ok=True)

    paths = [
        plot_overlay_behavior(os.path.join(out_dir, "overlay_behavior.png")),
        plot_agreement_residuals(os.path.join(out_dir, "agreement_residuals.png")),
        plot_derivatives(os.path.join(out_dir, "derivatives.png")),
    ]

    if verbose:
        for p in paths:
            print(os.path.abspath(p))

    return paths


class PlotSplinesTest(unittest.TestCase):
    """Generate the figures and assert the PNG files are created."""

    def test_generate_all_plots(self):
        paths = generate_all_plots(verbose=False)
        self.assertEqual(len(paths), 3)
        for p in paths:
            self.assertTrue(os.path.isfile(p), msg=f"missing figure: {p}")
            self.assertGreater(os.path.getsize(p), 0, msg=f"empty figure: {p}")


if __name__ == "__main__":
    generate_all_plots()

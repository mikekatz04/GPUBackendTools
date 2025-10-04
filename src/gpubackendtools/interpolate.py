# Interpolant for GPUs

# Copyright (C) 2021 Michael L. Katz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np

from .parallelbase import GBTParallelModuleBase
from .pointeradjust import wrapper


def searchsorted2d_vec(a, b, xp=None, **kwargs):
    if xp is None:
        xp = np

    m, n = a.shape
    max_num = xp.maximum(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num * xp.arange(a.shape[0])[:, None]
    p = xp.searchsorted((a + r).ravel(), (b + r).ravel(), **kwargs).reshape(m, -1)

    out = p - n * (xp.arange(m)[:, None])
    try:
        xp.cuda.runtime.deviceSynchronize()
    except AttributeError:
        pass

    return out


CUBIC_SPLINE_LINEAR_SPACING = 1
CUBIC_SPLINE_LOG10_SPACING = 2
CUBIC_SPLINE_GENERAL_SPACING = 3 


class CubicSplineInterpolant(GBTParallelModuleBase):
    """GPU-accelerated Multiple Cubic Splines

    This class produces multiple cubic splines. The cubic splines are produced
    with "not-a-knot" boundary conditions.

    This class has GPU capability.

    Args:
        x (xp.ndarray): f values as input for the spline. Can be 1D flattend array
            of total length
            ``(num_bin_all * length)`` or 2D array with shape: ``(num_bin_all, length)``.
        y_all (xp.ndarray): y values for the spline. This can be a 1D flattened
            array with length
            ``(num_interp_params * num_bin_all * num_modes * length)``
            or 4D arrays of shape: ``(num_interp_params, num_bin_all, num_modes, length)``.
        num_interp_params (int, optional): If ``x`` and ``y_all`` are flattened,
            the user must provide the number of interpolation parameters.
            (Default: ``None``)
        num_bin_all (int, optional): If ``x`` and ``y_all`` are flattened,
            the user must provide the number of total binaries.
            (Default: ``None``)
        num_modes (int, optional): If ``x`` and ``y_all`` are flattened,
            the user must provide the number of modes.
            (Default: ``None``)
        length (int, optional): If ``x`` and ``y_all`` are flattened,
            the user must provide the length of the frequency array for each binary.
            (Default: ``None``)
        force_backend (str, optional): ``"cpu"'', ``"gpu"'', ``"cuda"'', ``"cuda12x"'', or ``"cuda11x"''.

    Raises:
        ValueError: If input arguments are not correct.

    """

    def __init__(
        self,
        x,
        y_all,
        ninterps=None,
        length=None,
        spline_type=CUBIC_SPLINE_GENERAL_SPACING,
        force_backend=None,
    ):

        # check all inputs

        super().__init__(force_backend=force_backend)

        # first check is for flattened arrays
        if x.ndim == 1 or y_all.ndim == 1:
            if x.ndim != 1 or y_all.ndim != 1:
                raise ValueError(
                    "If providing flattened x and y_all, need to both be flattened."
                )
            if len(x) != len(y_all):
                raise ValueError("x and y must have same length.")
            
            if (
                length is None
                or ninterps is None
            ):
                raise ValueError(
                    "If providing flattened arrays, need to provide dimensional information: length, ninterps."
                )

            if len(x) != length * ninterps:
                raise ValueError(
                    f"Length of the x array is not correct. It is supposed to be {length * ninterps}. It is currently {len(x)}."
                )
            if len(y_all) != length * ninterps:
                raise ValueError(
                    f"Length of the y_all array is not correct. It is supposed to be {length * ninterps}. It is currently {len(y_all)}."
                )
            self.length = length
            self.ninterps = ninterps
            self.reshape_shape = (self.ninterps, self.length)

        else:
            # assumes last dimension is length
            if x.shape != y_all.shape:
                raise ValueError("x and y must have the same shape with the final dimension being the length of the spline.")
            
            # arrays are shaped
            self.reshape_shape = x.shape

            self.length = x.shape[-1]
            self.ninterps = int(np.prod(x.shape[:-1]))
            x = x.flatten()
            y_all = y_all.flatten()

        # get/store info
        self.degree = 3

        # setup all arrays for interpolation
        x_flat = self.xp.asarray(x)
        B_flat = self.xp.zeros((self.ninterps * self.length,))
        self.c1_flat = upper_diag = self.xp.zeros_like(B_flat)
        self.c2_flat = diag = self.xp.zeros_like(B_flat)
        self.c3_flat = lower_diag = self.xp.zeros_like(B_flat)
        self.y_flat = y_all
        self.x_flat = x.copy()

        self.spline_type = spline_type

        if spline_type == CUBIC_SPLINE_LINEAR_SPACING:
            assert self.xp.allclose(self.xp.diff(self.x_interp_shape, axis=-1), self.xp.diff(self.x_interp_shape, axis=-1)[:, 0][:, None])

        elif spline_type == CUBIC_SPLINE_LOG10_SPACING:
            assert self.xp.allclose(self.xp.diff(self.xp.log10(self.x_interp_shape), axis=-1), self.xp.diff(self.xp.log10(self.x_interp_shape), axis=-1)[:, 0][:, None])

        # perform interpolation
        self.interpolate_arrays(
            self.x_flat,
            self.y_flat,
            B_flat,
            upper_diag,
            diag,
            lower_diag,
            self.length,
            self.ninterps,
        )

        _inputs, tkwargs = wrapper(self.x_flat, self.y_flat, self.c1_flat, self.c2_flat, self.c3_flat, self.ninterps, self.length, self.spline_type)
        self._cpp_class = self.backend.pyCubicSplineWrap(*_inputs)

    @property
    def spline_type(self) -> int:
        return self._spline_type
    
    @spline_type.setter
    def spline_type(self, spline_type: int):
        if spline_type not in [CUBIC_SPLINE_LINEAR_SPACING, CUBIC_SPLINE_LOG10_SPACING, CUBIC_SPLINE_GENERAL_SPACING]:
            raise ValueError("spline_type must be one of CUBIC_SPLINE_LINEAR_SPACING, CUBIC_SPLINE_LOG10_SPACING, CUBIC_SPLINE_GENERAL_SPACING.")
        self._spline_type = spline_type

    @property
    def xp(self) -> object:
        """Numpy or Cupy"""
        return self.backend.xp
    
    @classmethod
    def supported_backends(cls) -> list:
        return ["gbt_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    @property
    def interpolate_arrays(self) -> callable:
        """C/CUDA wrapped function for computing interpolation."""
        return self.backend.interpolate_wrap

    @property
    def x(self):
        """Get shaped x array."""
        return self.x_flat.reshape(self.reshape_shape)
    
    @property
    def x_interp_shape(self):
        """Get shaped x array."""
        return self.x_flat.reshape(self.ninterps, self.length)

    @property
    def y(self):
        """Get shaped y array."""
        return self.y_flat.reshape(self.reshape_shape)
    
    @property
    def y_interp_shape(self):
        """Get shaped y array."""
        return self.y_flat.reshape(self.ninterps, self.length)

    @property
    def c1(self):
        """Get shaped c1 array."""
        return self.c1_flat.reshape(self.reshape_shape)

    @property
    def c1_interp_shape(self):
        """Get shaped c1 array."""
        return self.c1_flat.reshape(self.ninterps, self.length)
    
    @property
    def c2(self):
        """Get shaped c2 array."""
        return self.c2_flat.reshape(self.reshape_shape)

    @property
    def c2_interp_shape(self):
        """Get shaped c2 array."""
        return self.c2_flat.reshape(self.ninterps, self.length)
    
    @property
    def c3(self):
        """Get shaped c3 array."""
        return self.c3_flat.reshape(self.reshape_shape)

    @property
    def c3_interp_shape(self):
        """Get shaped c3 array."""
        return self.c3_flat.reshape(self.ninterps, self.length)
    
    @property
    def container(self):
        """Container for easy transit of interpolation information."""
        return [self.x_flat, self.y_flat, self.c1_flat, self.c2_flat, self.c3_flat]
    
    @property
    def cpp_class(self):
        return self._cpp_class
    
    def __call__(self, x_new, ind_interps = None, use_c_backend=False):
        
        if use_c_backend:
            raise NotImplementedError
        
        input_shape = x_new.shape
        if ind_interps is None:
            if not x_new.shape[:-1] == self.reshape_shape[:-1]:
                raise ValueError("Must add ind_interps if x_new is not same shape (except for the last axis) as the input x array.")
            
            ind_interps = self.xp.arange(self.ninterps)
            
        else:
            assert ind_interps.max().item() < self.ninterps
            assert ind_interps.min().item() >= 0
            if ind_interps.shape != x_new.shape[:-1]:
                raise ValueError("When inputing ind_interps, the shape needs to match x_new.shape[:-1].")
            ind_interps = ind_interps.flatten()

        assert len(ind_interps) == len(self.xp.unique(ind_interps))

        num_interps_here = len(ind_interps)

        ind_interps_all = self.xp.repeat(ind_interps[:, None], x_new.shape[-1], axis=-1).reshape(num_interps_here, x_new.shape[-1])
            
        x_new = x_new.reshape(num_interps_here, x_new.shape[-1])

        assert x_new.shape == ind_interps_all.shape
        assert self.xp.all(
            x_new <= self.x_interp_shape[ind_interps].max(axis=-1)[:, None]
        ) and self.xp.all(x_new >= self.x_interp_shape[ind_interps].min(axis=-1)[:, None])

        segment_inds = (
            searchsorted2d_vec(
                self.x_interp_shape[ind_interps],
                x_new.reshape(num_interps_here, x_new.shape[-1]),
                xp=self.xp,
                side="right",
            )
            - 1
        ).reshape(x_new.shape)

        if self.xp.any(segment_inds == self.length - 1):
            #  assert self.xp.all(x_new[segment_inds == self.length - 1] == self.x_shaped.max(axis=-1))
            segment_inds[segment_inds == self.length - 1] = self.length - 2

        x0 = self.x.reshape(self.ninterps, self.length)[ind_interps_all, segment_inds]
        y0 = self.y.reshape(self.ninterps, self.length)[ind_interps_all, segment_inds]

        c1 = self.c1.reshape(self.ninterps, self.length)[ind_interps_all, segment_inds]
        c2 = self.c2.reshape(self.ninterps, self.length)[ind_interps_all, segment_inds]
        c3 = self.c3.reshape(self.ninterps, self.length)[ind_interps_all, segment_inds]

        dx = x_new - x0

        y_new = y0 + c1 * dx + c2 * dx**2 + c3 * dx**3

        if hasattr(self.xp, "cuda"):
            self.xp.get_default_memory_pool().free_all_block()
        return y_new.reshape(input_shape)
    
    def interp_special(self, x_new, inds):
        assert x_new.ndim == 1
        assert x_new.shape[0] == inds.shape[0]

        assert self.xp.all(
            x_new <= self.x_shaped.reshape(-1, self.length)[inds].max(axis=-1)
        ) and self.xp.all(x_new >= self.x_shaped.reshape(-1, self.length)[inds].min(axis=-1))

        max_x = self.x_shaped.reshape(-1, self.length)[inds].max().item()
        scaled_x = ((self.x_shaped.reshape(-1, self.length)[np.unique(inds)] / max_x) + 100 * np.unique(inds)[:, None]).flatten()
        scaled_x_new = x_new / max_x + 100 * inds

        segment_inds = self.xp.searchsorted(scaled_x, scaled_x_new, side="right") - 1

        if self.xp.any(segment_inds == -1):
            #  assert self.xp.all(x_new[segment_inds == self.length - 1] == self.x_shaped.max(axis=-1))
            segment_inds[segment_inds == -1] = 0

        if self.xp.any(segment_inds == self.length - 1):
            #  assert self.xp.all(x_new[segment_inds == self.length - 1] == self.x_shaped.max(axis=-1))
            segment_inds[segment_inds == self.length - 1] = self.length - 2

        x0 = self.x_shaped.reshape(-1, self.length)[(inds, segment_inds)]
        
        _inds_y0 = self.xp.repeat(inds, self.y_shaped.shape[0])
        _segment_inds_y0 = self.xp.repeat(segment_inds, self.y_shaped.shape[0])
        _interp_params_inds = self.xp.tile(self.xp.arange(self.y_shaped.shape[0]), (len(inds),))
        y0 = self.y_shaped.reshape(self.y_shaped.shape[0], -1, self.length)[_interp_params_inds, _inds_y0, _segment_inds_y0].reshape(self.y_shaped.shape[0], -1)
        c1 = self.c1_shaped.reshape(self.y_shaped.shape[0], -1, self.length)[_interp_params_inds, _inds_y0, _segment_inds_y0].reshape(self.y_shaped.shape[0], -1)
        c2 = self.c2_shaped.reshape(self.y_shaped.shape[0], -1, self.length)[_interp_params_inds, _inds_y0, _segment_inds_y0].reshape(self.y_shaped.shape[0], -1)
        c3 = self.c3_shaped.reshape(self.y_shaped.shape[0], -1, self.length)[_interp_params_inds, _inds_y0, _segment_inds_y0].reshape(self.y_shaped.shape[0], -1)

        dx = x_new - x0
        y_new = y0 + c1 * dx[None, :] + c2 * dx[None, :] **2 + c3 * dx[None, :]**3

        if hasattr(self.xp, "cuda"):
            self.xp.get_default_memory_pool().free_all_block()
        return y_new

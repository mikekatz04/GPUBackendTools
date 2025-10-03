import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uintptr_t

from gpubackendtools import wrapper

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "Interpolate.hh":
    void interpolate(double* x, double* propArrays,
                     double* B, double* upper_diag, double* diag, double* lower_diag,
                     int length, int ninterps);

    cdef cppclass CubicSplineWrap "CubicSpline":
        CubicSplineWrap(double *x0_, double *y0_, double *c1_, double *c2_, double *c3_, double ninterps_, int length_, int spline_type_) except+
        void eval(double *y_new, double *x_new, int* spline_index, int N) except+
        double eval_single(double x_new, int spline_index) except+
        void dealloc() except+

    
cdef class pyCubicSplineWrap:
    cdef CubicSplineWrap *g
    cdef uintptr_t x0_ptr
    cdef uintptr_t y0_ptr
    cdef uintptr_t c1_ptr
    cdef uintptr_t c2_ptr
    cdef uintptr_t c3_ptr
    cdef int length
    cdef int ninterps
    cdef int spline_type


    def __cinit__(self, x0, y0, c1, c2, c3, ninterps, length, spline_type):
        self.x0_ptr = x0
        self.y0_ptr = y0
        self.c1_ptr = c1
        self.c2_ptr = c2
        self.c3_ptr = c3
        self.ninterps = ninterps
        self.length = length
        self.spline_type = spline_type

        cdef size_t x0_in = x0
        cdef size_t y0_in = y0
        cdef size_t c1_in = c1
        cdef size_t c2_in = c2
        cdef size_t c3_in = c3
        
        self.g = new CubicSplineWrap(<double *>x0_in, <double *>y0_in, <double *>c1_in, <double *>c2_in, <double *>c3_in, ninterps, length, spline_type)

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g

    def __reduce__(self):
        return (rebuild_cublic_spline, (self.x0_ptr, self.y0_prt, self.c1_ptr, self.c2_ptr, self.c3_ptr, self.ninterps, self.length, self.spline_type))

    @property
    def ptr(self) -> long:
        return <uintptr_t>self.g

    def eval_single(self, x_new, spline_index):
        return self.g.eval_single(x_new, spline_index)

    def eval(self, *args, **kwargs):
        (y_new, x_new, spline_index, N), tkwargs = wrapper(*args, **kwargs)
        cdef size_t x_new_in = x_new
        cdef size_t y_new_in = y_new
        cdef size_t spline_index_in = spline_index
        self.g.eval(<double*> y_new_in, <double*> x_new_in, <int*>spline_index_in, N)

def rebuild_cublic_spline(x0_ptr, y0_ptr, c1_ptr, c2_ptr, c3_ptr, ninterps, length, spline_type):
    c = pyCubicSplineWrap(x0_ptr, y0_ptr, c1_ptr, c2_ptr, c3_ptr, ninterps, length, spline_type)
    return c

def interpolate_wrap(*args, **kwargs):

    (
        x, propArrays,
        B, upper_diag, diag, lower_diag,
        length, ninterps
    ), tkwargs = wrapper(*args, **kwargs)

    cdef size_t x_in = x
    cdef size_t propArrays_in = propArrays
    cdef size_t B_in = B
    cdef size_t upper_diag_in = upper_diag
    cdef size_t diag_in = diag
    cdef size_t lower_diag_in = lower_diag

    interpolate(<double*>x_in, <double*>propArrays_in,
              <double*>B_in, <double*>upper_diag_in, <double*>diag_in, <double*>lower_diag_in,
              length, ninterps)

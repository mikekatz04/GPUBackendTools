#ifndef __INTERPOLATE_HH__
#define __INTERPOLATE_HH__

#include "gbt_global.h"
// The CubicSpline / CubicSplineSegment classes + the `CUBIC_SPLINE_*`
// spacing constants are now header-only in InterpolateDevice.hh so that
// downstream `.cu` files can include them and evaluate splines without
// linking against GBT's `Interpolate.cu` translation unit. The host-side
// build/solve (`interpolate`, `eval_wrap`, `fit_cubic_spline_*`) lives
// here and is built into GBT's plugin wheel.
#include "InterpolateDevice.hh"

void interpolate(double* x, double* propArrays,
                 double* B, double* upper_diag, double* diag, double* lower_diag,
                 int length, int ninterps);

void eval_wrap(CubicSpline *spline, double *y_new, double *x_new, int *spline_index, int N);

// ---- Quintic (degree-5, not-a-knot) spline: batched build/solve ----
// Reproduces scipy.interpolate.make_interp_spline(x, y, k=5) (bc_type=None) at
// the result level (~machine precision). Fills the five power-basis coefficient
// arrays c1..c5 (each of size ninterps*length, flattened interp_i*length + i)
// in place from (x, y, also ninterps*length flattened). Each spline must have
// length >= 6 and a strictly increasing x grid. The banded-collocation scratch
// is allocated internally (see Interpolate.cu); callers provide only c1..c5.
void interpolate_quintic(double* x, double* y,
                         double* c1, double* c2, double* c3, double* c4, double* c5,
                         int length, int ninterps);

void eval_quintic_wrap(QuinticSpline *spline, double *y_new, double *x_new, int *spline_index, int N);

#if !defined(__CUDA_COMPILATION__) && !defined(__CUDACC__)
// (CPU-only) Fit cubic-spline coefficients (c1, c2, c3) for ONE spline of
// length `length` from (x, y) data using the Thomas algorithm to solve the
// not-a-knot tridiagonal system. Returns a CubicSpline (with ninterps = 1)
// pointing at the now-filled buffers.
//
// All buffers (x, y, c1, c2, c3, B) must have size `length` and live as long
// as the returned CubicSpline. B is overwritten with the first-derivative
// solution vector; c1, c2, c3 may be uninitialized on entry — they are
// repurposed as the upper / main / lower diagonals during the solve and then
// overwritten with the final spline coefficients.
//
// Thomas is in-place and pivot-free, so it is faster than the LAPACKE_dgtsv
// path used by interpolate(...) but only stable when the system is
// diagonally dominant. The not-a-knot interior rows are; the first and last
// rows are not, so callers should validate accuracy on their input grids.
CubicSpline fit_cubic_spline_thomas(double *x, double *y,
                                    double *c1, double *c2, double *c3,
                                    double *B,
                                    int length, int spline_type);

// (CPU-only) Host launcher exposed to Python; thin wrapper around
// fit_cubic_spline_thomas.
void fit_cubic_spline_thomas_run(double *x, double *y,
                                 double *c1, double *c2, double *c3,
                                 double *B,
                                 int length, int spline_type);
#endif

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
// (GPU-only) Host-callable launcher for fit_cubic_spline_pcr. Launches a
// single block whose threads cooperate on the PCR sweep, with `8 * length`
// doubles of dynamic shared memory.
void fit_cubic_spline_pcr_run(double *x, double *y,
                              double *c1, double *c2, double *c3,
                              double *B,
                              int length, int spline_type);
#endif

#ifdef __CUDACC__

// (GPU-only) Same contract as fit_cubic_spline_thomas, but solves the
// not-a-knot tridiagonal system using Parallel Cyclic Reduction (PCR) to
// exploit thread-level parallelism within a block. Intended to be called
// from a kernel launched with one block per spline; threads of the block
// cooperate on the PCR sweep and synchronize via __syncthreads(). All
// threads of the block must enter this function with identical arguments —
// it issues __syncthreads() between stages.
//
// pcr_scratch is caller-allocated working space of at least 8 * length
// doubles, used as four (a, b, c, d) ping-pong pairs. Typical use sizes it
// dynamically as shared memory at the kernel launch:
//
//     extern __shared__ double smem[];
//     ...
//     fit_cubic_spline_pcr(..., smem, length, spline_type);
//
// launched with `kernel<<<grid, block, 8 * length * sizeof(double)>>>(...)`.
// Global-memory scratch is also accepted (correct but slower).
CUDA_DEVICE
CubicSpline fit_cubic_spline_pcr(double *x, double *y,
                                 double *c1, double *c2, double *c3,
                                 double *B,
                                 double *pcr_scratch,
                                 int length, int spline_type);

#endif // __CUDACC__

#endif // __INTERPOLATE_HH__

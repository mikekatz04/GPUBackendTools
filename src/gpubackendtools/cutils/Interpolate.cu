#include "gbt_global.h"
#include "Interpolate.hh"

#ifdef __CUDACC__
#include "cusparse_v2.h"
#else
#include "lapacke.h"
#endif

#define NUM_THREADS_INTERPOLATE 256


// See scipy CubicSpline implementation, it matches that
CUDA_DEVICE
void prep_splines(int i, int length, int interp_i, int ninterps, double *b, double *ud, double *diag, double *ld, double *x, double *y)
{
  double dx1, dx2, d, slope1, slope2;
  int ind0x, ind1x, ind2x, ind0y, ind1y, ind2y, ind_out;

  double xval0, xval1, xval2, yval1;

  // get proper frequency array since it is given once for all modes
  
  // fill values in spline initial computations
  // get indices into the 1D arrays
  // compute necessary quantities
  // fill the diagonals
  if (i == length - 1)
  {

    ind0y = interp_i * length + (length - 3);
    ind1y = interp_i * length + (length - 2);
    ind2y = interp_i * length + (length - 1);

    ind0x = interp_i * length + (length - 3);
    ind1x = interp_i * length + (length - 2);
    ind2x = interp_i * length + (length - 1);

    ind_out = interp_i * length + (length - 1);

    xval0 = x[ind0x];
    xval1 = x[ind1x];
    xval2 = x[ind2x];

    dx1 = xval1 - xval0;
    dx2 = xval2 - xval1;
    d = xval2 - xval0;

    yval1 = y[ind1y];

    slope1 = (yval1 - y[ind0y]) / dx1;
    slope2 = (y[ind2y] - yval1) / dx2;

    b[ind_out] = ((dx2 * dx2 * slope1 +
                   (2 * d + dx2) * dx1 * slope2) /
                  d);
    diag[ind_out] = dx1;
    ld[ind_out] = d;
    ud[ind_out] = 0.0;
  }
  else if (i == 0)
  {

    ind0y = interp_i * length + 0;
    ind1y = interp_i * length + 1;
    ind2y = interp_i * length + 2;

    ind0x = interp_i * length + 0;
    ind1x = interp_i * length + 1;
    ind2x = interp_i * length + 2;

    ind_out = interp_i * length + 0;

    xval0 = x[ind0x];
    xval1 = x[ind1x];
    xval2 = x[ind2x];

    dx1 = xval1 - xval0;
    dx2 = xval2 - xval1;
    d = xval2 - xval0;

    yval1 = y[ind1y];

    // amp
    slope1 = (yval1 - y[ind0y]) / dx1;
    slope2 = (y[ind2y] - yval1) / dx2;

    b[ind_out] = ((dx1 + 2 * d) * dx2 * slope1 +
                  dx1 * dx1 * slope2) /
                 d;
    ud[ind_out] = d;
    ld[ind_out] = 0.0;
    diag[ind_out] = dx2;
  }
  else
  {

    ind0y = interp_i * length + (i - 1);
    ind1y = interp_i * length + (i + 0);
    ind2y = interp_i * length + (i + 1);

    ind0x = interp_i * length + (i - 1);
    ind1x = interp_i * length + (i + 0);
    ind2x = interp_i * length + (i + 1);

    ind_out = interp_i * length + i;

    xval0 = x[ind0x];
    xval1 = x[ind1x];
    xval2 = x[ind2x];

    dx1 = xval1 - xval0;
    dx2 = xval2 - xval1;

    yval1 = y[ind1y];

    // amp
    slope1 = (yval1 - y[ind0y]) / dx1;
    slope2 = (y[ind2y] - yval1) / dx2;

    b[ind_out] = 3.0 * (dx2 * slope1 + dx1 * slope2);
    diag[ind_out] = 2 * (dx1 + dx2);
    ud[ind_out] = dx1;
    ld[ind_out] = dx2;
  }
}

CUDA_KERNEL
void fill_B(double *freqs_arr, double *y_all, double *B, double *upper_diag, double *diag, double *lower_diag,
            int ninterps, int length)
{

#ifdef __CUDACC__

  int start1 = blockIdx.x;
  int end1 = ninterps;
  int diff1 = gridDim.x;

#else

  int start1 = 0;
  int end1 = ninterps;
  int diff1 = 1;

#endif
  for (int interp_i = start1;
       interp_i < end1; // 2 for re and im
       interp_i += diff1)
  {

#ifdef __CUDACC__

    int start2 = threadIdx.x;
    int end2 = length;
    int diff2 = blockDim.x;

#else

    int start2 = 0;
    int end2 = length;
    int diff2 = 1;

#endif
    for (int i = start2;
         i < end2;
         i += diff2)
    {
        prep_splines(i, length, interp_i, ninterps, B, upper_diag, diag, lower_diag, freqs_arr, y_all);
    }
  }
}

/*
CuSparse error checking
*/
#define ERR_NE(X, Y)                                                           \
  do                                                                           \
  {                                                                            \
    if ((X) != (Y))                                                            \
    {                                                                          \
      fprintf(stderr, "Error in %s at %s:%d\n", __func__, __FILE__, __LINE__); \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define CUSPARSE_CALL(X) ERR_NE((X), CUSPARSE_STATUS_SUCCESS)

// See scipy CubicSpline implementation, it matches that
// this is for solving the banded matrix equation
void interpolate_kern(int m, int n, double *a, double *b, double *c, double *d_in)
{
#ifdef __CUDACC__
  size_t bufferSizeInBytes;

  cusparseHandle_t handle;
  void *pBuffer;

  CUSPARSE_CALL(cusparseCreate(&handle));
  CUSPARSE_CALL(cusparseDgtsv2StridedBatch_bufferSizeExt(handle, m, a, b, c, d_in, n, m, &bufferSizeInBytes));
  gpuErrchk(cudaMalloc(&pBuffer, bufferSizeInBytes));

  CUSPARSE_CALL(cusparseDgtsv2StridedBatch(handle,
                                           m,
                                           a, // dl
                                           b, // diag
                                           c, // du
                                           d_in,
                                           n,
                                           m,
                                           pBuffer));

  CUSPARSE_CALL(cusparseDestroy(handle));
  gpuErrchk(cudaFree(pBuffer));

#else

// use lapack on CPU
for (int j = 0;
      j < n;
      j += 1)
{
  int info = LAPACKE_dgtsv(LAPACK_COL_MAJOR, m, 1, &a[j * m + 1], &b[j * m], &c[j * m], &d_in[j * m], m);
  // if (info != m) printf("lapack info check: %d\n", info);
}

#endif
}

// See Scipy CubicSpline for more information
CUDA_DEVICE
void fill_coefficients(int i, int length, int interp_i, int ninterps, double *dydx, double *x, double *y, double *coeff1, double *coeff2, double *coeff3)
{
  double slope, t, dydx_i;

  int ind_i = interp_i * length + i;
  int ind_ip1 = interp_i * length + (i + 1);

  double dx = x[ind_ip1] - x[ind_i];

  slope = (y[ind_ip1] - y[ind_i]) / dx;

  dydx_i = dydx[ind_i];

  t = (dydx_i + dydx[ind_ip1] - 2 * slope) / dx;

  coeff1[ind_i] = dydx_i;
  coeff2[ind_i] = (slope - dydx_i) / dx - t;
  coeff3[ind_i] = t / dx;
}

CUDA_KERNEL
void set_spline_constants(double *x, double *y, double *c1, double *c2, double *c3, double *B,
                          int ninterps, int length)
{

  double df;
#ifdef __CUDACC__
  int start1 = blockIdx.x;
  int end1 = ninterps;
  int diff1 = gridDim.x;
#else

  int start1 = 0;
  int end1 = ninterps;
  int diff1 = 1;

#endif

  for (int interp_i = start1;
       interp_i < end1; // 2 for re and im
       interp_i += diff1)
  {
    // int freqArr_i = sub_i; // int(sub_i / num_intermediates);

#ifdef __CUDACC__
    int start2 = threadIdx.x;
    int end2 = length - 1;
    int diff2 = blockDim.x;
#else

    int start2 = 0;
    int end2 = length - 1;
    int diff2 = 1;

#endif
    for (int i = start2;
         i < end2;
         i += diff2)
    {

      int lead_ind = interp_i * length;
      fill_coefficients(i, length, interp_i, ninterps, B, x,
                        y,
                        c1,
                        c2,
                        c3);
    }
  }
}

void interpolate(double *x, double *propArrays,
                 double *B, double *upper_diag, double *diag, double *lower_diag,
                 int length, int ninterps)
{

  int nblocks = std::ceil((ninterps + NUM_THREADS_INTERPOLATE - 1) / NUM_THREADS_INTERPOLATE);

  // these are used for both coefficients and diagonals because they are the same size and
  // this reduces the total memory needed
  double *c1 = upper_diag;
  double *c2 = diag;
  double *c3 = lower_diag;

  // process is fill the B matrix which is banded.
  // solve banded matrix equation for spline coefficients
  // Fill the spline coefficients properly

#ifdef __CUDACC__
  fill_B<<<nblocks, NUM_THREADS_INTERPOLATE>>>(x, propArrays, B, upper_diag, diag, lower_diag, ninterps, length);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  interpolate_kern(length, ninterps, lower_diag, diag, upper_diag, B);

  set_spline_constants<<<nblocks, NUM_THREADS_INTERPOLATE>>>(x, propArrays, c1, c2, c3, B,
                                                             ninterps, length);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
#else
  fill_B(x, propArrays, B, upper_diag, diag, lower_diag, ninterps, length);

  interpolate_kern(length, ninterps, lower_diag, diag, upper_diag, B);

  set_spline_constants(x, propArrays, c1, c2, c3, B,
                       ninterps, length);
#endif
}


// CubicSpline::get_window / get_cublic_spline_segment / eval_single / eval
// were moved to InterpolateDevice.hh as inline device methods so downstream
// `.cu` translation units can evaluate splines without linking against this
// archive. The host-launcher path (`eval_kernel` + `eval_wrap`) below still
// lives here.

CUDA_KERNEL
void eval_kernel(CubicSpline *spline, double *y_new, double *x_new, int *spline_index, int N)
{
    spline->eval(y_new, x_new, spline_index, N);
}

void eval_wrap(CubicSpline *spline, double *y_new, double *x_new, int *spline_index, int N)
{
#ifdef __CUDACC__
    int nblocks = std::ceil((N + NUM_THREADS_INTERPOLATE - 1) / NUM_THREADS_INTERPOLATE);

    // copy this class to device
    CubicSpline *d_spline;
    gpuErrchk(cudaMalloc(&d_spline, sizeof(CubicSpline)));
    gpuErrchk(cudaMemcpy(d_spline, spline, sizeof(CubicSpline), cudaMemcpyHostToDevice));

    eval_kernel<<<nblocks, NUM_THREADS_INTERPOLATE>>>(d_spline, y_new, x_new, spline_index, N);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(d_spline));

#else
    spline->eval(y_new, x_new, spline_index, N);
#endif
}

#ifndef __CUDACC__

CubicSpline fit_cubic_spline_thomas(double *x, double *y,
                                    double *c1, double *c2, double *c3,
                                    double *B,
                                    int length, int spline_type)
{
    // 1. Fill the not-a-knot tridiagonal system: B is the rhs, c1 / c2 / c3
    //    double as upper / main / lower diagonals. prep_splines uses
    //    interp_i*length + i indexing; with interp_i = 0 the offsets reduce
    //    to plain `i`, so we treat the inputs as a single-row layout.
    for (int i = 0; i < length; ++i)
    {
        prep_splines(i, length, /*interp_i=*/0, /*ninterps=*/1,
                     B, c1, c2, c3, x, y);
    }

    // 2. Thomas tridiagonal sweep, in place. c2 is the working main diagonal;
    //    B becomes the dy/dx solution vector on exit.
    for (int i = 1; i < length; ++i)
    {
        double w = c3[i] / c2[i - 1];
        c2[i] -= w * c1[i - 1];
        B[i]  -= w * B[i - 1];
    }
    B[length - 1] /= c2[length - 1];
    for (int i = length - 2; i >= 0; --i)
    {
        B[i] = (B[i] - c1[i] * B[i + 1]) / c2[i];
    }

    // 3. Convert derivatives in B into the final spline coefficients. This
    //    overwrites c1, c2, c3 — the diagonals are no longer needed.
    for (int i = 0; i < length - 1; ++i)
    {
        fill_coefficients(i, length, /*interp_i=*/0, /*ninterps=*/1,
                          B, x, y, c1, c2, c3);
    }

    return CubicSpline(x, y, c1, c2, c3, /*ninterps=*/1, length, spline_type);
}

void fit_cubic_spline_thomas_run(double *x, double *y,
                                 double *c1, double *c2, double *c3,
                                 double *B,
                                 int length, int spline_type)
{
    fit_cubic_spline_thomas(x, y, c1, c2, c3, B, length, spline_type);
}

#endif // !__CUDACC__

#ifdef __CUDACC__

CUDA_KERNEL
void fit_cubic_spline_pcr_launch(double *x, double *y,
                                 double *c1, double *c2, double *c3,
                                 double *B,
                                 int length, int spline_type)
{
    extern __shared__ double pcr_smem[];
    fit_cubic_spline_pcr(x, y, c1, c2, c3, B, pcr_smem, length, spline_type);
}

void fit_cubic_spline_pcr_run(double *x, double *y,
                              double *c1, double *c2, double *c3,
                              double *B,
                              int length, int spline_type)
{
    int threads = (length < NUM_THREADS_INTERPOLATE) ? length : NUM_THREADS_INTERPOLATE;
    if (threads < 1) threads = 1;
    size_t shared_bytes = (size_t)8 * (size_t)length * sizeof(double);
    fit_cubic_spline_pcr_launch<<<1, threads, shared_bytes>>>(x, y, c1, c2, c3, B, length, spline_type);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

CUDA_DEVICE
CubicSpline fit_cubic_spline_pcr(double *x, double *y,
                                 double *c1, double *c2, double *c3,
                                 double *B,
                                 double *pcr_scratch,
                                 int length, int spline_type)
{
    // Lay out the caller-provided scratch as four (a, b, c, d) ping-pong
    // pairs. Each "buffer" is `length` doubles; src indexes the current
    // state, dst receives the next reduction step, and we swap each
    // iteration.
    double *a_buf[2] = { pcr_scratch + 0 * length, pcr_scratch + 1 * length };
    double *b_buf[2] = { pcr_scratch + 2 * length, pcr_scratch + 3 * length };
    double *c_buf[2] = { pcr_scratch + 4 * length, pcr_scratch + 5 * length };
    double *d_buf[2] = { pcr_scratch + 6 * length, pcr_scratch + 7 * length };

    // 1. Build the tridiagonal system. prep_splines writes the rhs into B
    //    and the upper / main / lower diagonals into c1 / c2 / c3 (with
    //    interp_i = 0, ninterps = 1 the offsets reduce to plain `i`).
    for (int i = threadIdx.x; i < length; i += blockDim.x)
    {
        prep_splines(i, length, /*interp_i=*/0, /*ninterps=*/1,
                     B, c1, c2, c3, x, y);
    }
    __syncthreads();

    // 2. Stage into the scratch ping-pong. a = c3 (lower), b = c2 (diag),
    //    c = c1 (upper), d = B (rhs).
    int src = 0, dst = 1;
    for (int i = threadIdx.x; i < length; i += blockDim.x)
    {
        a_buf[src][i] = c3[i];
        b_buf[src][i] = c2[i];
        c_buf[src][i] = c1[i];
        d_buf[src][i] = B[i];
    }
    __syncthreads();

    // 3. PCR sweep. At step k each row i is recombined with rows i ± stride
    //    (stride = 2^(k-1)) to eliminate x_{i±stride}. Stride doubles each
    //    iteration; once stride >= length both neighbors are out of bounds
    //    and the system is fully decoupled, giving x_i = d_i / b_i.
    for (int stride = 1; stride < length; stride <<= 1)
    {
        for (int i = threadIdx.x; i < length; i += blockDim.x)
        {
            double a_i = a_buf[src][i];
            double b_i = b_buf[src][i];
            double c_i = c_buf[src][i];
            double d_i = d_buf[src][i];

            int im = i - stride;
            int ip = i + stride;

            double alpha = 0.0, beta = 0.0;
            double am = 0.0, cm = 0.0, dm = 0.0;
            double ap = 0.0, cp = 0.0, dp = 0.0;

            if (im >= 0)
            {
                am = a_buf[src][im];
                cm = c_buf[src][im];
                dm = d_buf[src][im];
                alpha = -a_i / b_buf[src][im];
            }
            if (ip < length)
            {
                ap = a_buf[src][ip];
                cp = c_buf[src][ip];
                dp = d_buf[src][ip];
                beta = -c_i / b_buf[src][ip];
            }

            a_buf[dst][i] = alpha * am;
            b_buf[dst][i] = b_i + alpha * cm + beta * ap;
            c_buf[dst][i] = beta * cp;
            d_buf[dst][i] = d_i + alpha * dm + beta * dp;
        }
        __syncthreads();

        int tmp = src;
        src = dst;
        dst = tmp;
    }

    // 4. Decoupled solve. Write dy/dx solution back to B.
    for (int i = threadIdx.x; i < length; i += blockDim.x)
    {
        B[i] = d_buf[src][i] / b_buf[src][i];
    }
    __syncthreads();

    // 5. Convert derivatives in B into the final spline coefficients.
    for (int i = threadIdx.x; i < length - 1; i += blockDim.x)
    {
        fill_coefficients(i, length, /*interp_i=*/0, /*ninterps=*/1,
                          B, x, y, c1, c2, c3);
    }
    __syncthreads();

    return CubicSpline(x, y, c1, c2, c3, /*ninterps=*/1, length, spline_type);
}

#endif // __CUDACC__

// ===========================================================================
// Quintic (degree-5, not-a-knot) spline build/solve.
//
// Reproduces scipy.interpolate.make_interp_spline(x, y, k=5) (bc_type=None) at
// the result level. Per spline we build the degree-5 B-spline collocation
// system A c = y on the not-a-knot knot vector, solve the (4,4)-banded system
// with de Boor's pivot-free banded elimination (banfac/banslv -- stable because
// the collocation matrix is totally positive), then differentiate the B-spline
// at each left breakpoint to obtain the per-segment power-basis coefficients
// c1..c5 (Taylor coefficients a_k = s^(k)(x_i)/k!, with y0 = y_i).
//
// Structure mirrors the cubic: fill_quintic_band (build) / solve (banded LU) /
// set_quintic_constants (extract), so the same .cu compiles for CPU and GPU.
// Validated against scipy to <=1.2e-13 across n=6..1000, equal/log/random grids.
// ===========================================================================

#define QUINTIC_DEG 5
#define QUINTIC_HALF_BAND 4              // true nonzero half-bandwidth (kl=ku)
#define QUINTIC_BAND_ROWS 9              // 2*QUINTIC_HALF_BAND + 1

// interp_i-innermost (coalesced) layout for the internal solve scratch. The
// band W and rhs/coef B are per-call scratch private to interpolate_quintic;
// laying interp_i innermost makes the memory-bound one-thread-per-spline solve
// coalesce (neighbouring threads -> neighbouring addresses) for any length, no
// shared memory, no algorithm change. The OUTPUT c1..c5 keep interp_i*length+i.
//   band element (band_row b in [0,8], col c in [0,length)) of spline interp_i
//   rhs/coef element (col c) of spline interp_i
#define QBAND(W, b, c, interp_i, ninterps, length) \
    (W)[(((size_t)(b) * (length) + (c)) * (ninterps)) + (interp_i)]
#define QRHS(B, c, interp_i, ninterps) \
    (B)[((size_t)(c) * (ninterps)) + (interp_i)]

// Closed-form not-a-knot knot vector entry t[k], k in [0, n+5], for the grid x
// of length n. t = [x0]*6 ++ x[3:n-3] ++ [x_{n-1}]*6. No materialization needed.
CUDA_DEVICE
double quintic_knot(double *x, int n, int k)
{
    if (k <= 5) return x[0];
    if (k >= n) return x[n - 1];
    return x[k - 3];
}

// Half-open knot span l with t[l] <= xv < t[l+1] (NURBS A2.1); right end -> n-1.
CUDA_DEVICE
int quintic_find_span(double *x, int n, double xv)
{
    int last = n - 1;
    if (xv >= quintic_knot(x, n, last + 1)) return last;
    int low = QUINTIC_DEG;
    int high = last + 1;
    int mid = (low + high) / 2;
    while (xv < quintic_knot(x, n, mid) || xv >= quintic_knot(x, n, mid + 1))
    {
        if (xv < quintic_knot(x, n, mid))
            high = mid;
        else
            low = mid;
        mid = (low + high) / 2;
    }
    return mid;
}

// NURBS A2.2: the 6 nonzero degree-5 basis values N[0..5] = B_{l-5+j,5}(xv).
CUDA_DEVICE
void quintic_basis_funs(double *x, int n, int l, double xv, double *N)
{
    double left[QUINTIC_DEG + 1];
    double right[QUINTIC_DEG + 1];
    N[0] = 1.0;
    for (int j = 1; j <= QUINTIC_DEG; ++j)
    {
        left[j] = xv - quintic_knot(x, n, l + 1 - j);
        right[j] = quintic_knot(x, n, l + j) - xv;
        double saved = 0.0;
        for (int r = 0; r < j; ++r)
        {
            double temp = N[r] / (right[r + 1] + left[j - r]);
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        N[j] = saved;
    }
}

// NURBS A2.3: ders[k][j] = k-th derivative (k=0..5) of basis B_{l-5+j,5}(xv).
CUDA_DEVICE
void quintic_ders_basis_funs(double *x, int n, int l, double xv, double ders[QUINTIC_DEG + 1][QUINTIC_DEG + 1])
{
    double ndu[QUINTIC_DEG + 1][QUINTIC_DEG + 1];
    double a[2][QUINTIC_DEG + 1];
    double left[QUINTIC_DEG + 1];
    double right[QUINTIC_DEG + 1];

    ndu[0][0] = 1.0;
    for (int j = 1; j <= QUINTIC_DEG; ++j)
    {
        left[j] = xv - quintic_knot(x, n, l + 1 - j);
        right[j] = quintic_knot(x, n, l + j) - xv;
        double saved = 0.0;
        for (int r = 0; r < j; ++r)
        {
            ndu[j][r] = right[r + 1] + left[j - r];
            double temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    for (int j = 0; j <= QUINTIC_DEG; ++j) ders[0][j] = ndu[j][QUINTIC_DEG];

    for (int r = 0; r <= QUINTIC_DEG; ++r)
    {
        int s1 = 0;
        int s2 = 1;
        a[0][0] = 1.0;
        for (int k = 1; k <= QUINTIC_DEG; ++k)
        {
            double d = 0.0;
            int rk = r - k;
            int pk = QUINTIC_DEG - k;
            if (r >= k)
            {
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                d = a[s2][0] * ndu[rk][pk];
            }
            int j1 = (rk >= -1) ? 1 : -rk;
            int j2 = ((r - 1) <= pk) ? (k - 1) : (QUINTIC_DEG - r);
            for (int j = j1; j <= j2; ++j)
            {
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
                d += a[s2][j] * ndu[rk + j][pk];
            }
            if (r <= pk)
            {
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                d += a[s2][k] * ndu[r][pk];
            }
            ders[k][r] = d;
            int tmp = s1;
            s1 = s2;
            s2 = tmp;
        }
    }
    // Multiply through by the factorial factors p!/(p-k)!.
    int fac = QUINTIC_DEG;
    for (int k = 1; k <= QUINTIC_DEG; ++k)
    {
        for (int j = 0; j <= QUINTIC_DEG; ++j) ders[k][j] *= fac;
        fac *= (QUINTIC_DEG - k);
    }
}

// Build collocation row i (point x_i) into spline interp_i's banded scratch +
// rhs. Band/rhs use the interp_i-innermost layout (QBAND/QRHS): A[i,j] stored at
// QBAND(W, QUINTIC_HALF_BAND + i - j, j, interp_i, ninterps, length). W must be
// pre-zeroed. (Inputs x, y keep their interp_i*length+i layout.)
CUDA_DEVICE
void prep_quintic_band(int i, int length, int interp_i, int ninterps,
                       double *W, double *B, double *x, double *y)
{
    double *xs = &x[interp_i * length];

    int l = quintic_find_span(xs, length, xs[i]);
    double N[QUINTIC_DEG + 1];
    quintic_basis_funs(xs, length, l, xs[i], N);
    for (int jj = 0; jj <= QUINTIC_DEG; ++jj)
    {
        int j = l - QUINTIC_DEG + jj;
        int off = i - j;
        // Structural zeros (de Boor offset 5 at the right end) fall outside the
        // 4-band; skip them -- W is already zeroed.
        if (off >= -QUINTIC_HALF_BAND && off <= QUINTIC_HALF_BAND)
            QBAND(W, QUINTIC_HALF_BAND + off, j, interp_i, ninterps, length) = N[jj];
    }
    QRHS(B, i, interp_i, ninterps) = y[interp_i * length + i];
}

// Pivot-free banded LU (de Boor banfac), half-bandwidth 4, in place on spline
// interp_i's band (interp_i-innermost layout, n = length). Returns 0 on success,
// (row+1) on a zero pivot (never for a valid strictly-increasing grid: the
// collocation matrix is totally positive).
CUDA_DEVICE
int quintic_banfac(double *W, int interp_i, int ninterps, int n)
{
    const int m = QUINTIC_HALF_BAND;
    for (int i = 0; i < n; ++i)
    {
        double piv = QBAND(W, m, i, interp_i, ninterps, n);
        if (piv == 0.0) return i + 1;
        int imax = (m < (n - 1 - i)) ? m : (n - 1 - i);
        for (int s = 1; s <= imax; ++s)
        {
            double fac = QBAND(W, m + s, i, interp_i, ninterps, n) / piv;
            QBAND(W, m + s, i, interp_i, ninterps, n) = fac;
            for (int r = 1; r <= imax; ++r)
                QBAND(W, m + s - r, i + r, interp_i, ninterps, n) -=
                    fac * QBAND(W, m - r, i + r, interp_i, ninterps, n);
        }
    }
    return 0;
}

// Solve after quintic_banfac; B (interp_i-innermost rhs, n = length) overwritten
// with the B-spline coefficients for spline interp_i.
CUDA_DEVICE
void quintic_banslv(double *W, double *B, int interp_i, int ninterps, int n)
{
    const int m = QUINTIC_HALF_BAND;
    for (int i = 0; i < n; ++i)                 // forward (unit lower L)
    {
        int imax = (m < (n - 1 - i)) ? m : (n - 1 - i);
        for (int s = 1; s <= imax; ++s)
            QRHS(B, i + s, interp_i, ninterps) -=
                QBAND(W, m + s, i, interp_i, ninterps, n) * QRHS(B, i, interp_i, ninterps);
    }
    for (int i = n - 1; i >= 0; --i)            // back (upper U)
    {
        QRHS(B, i, interp_i, ninterps) /= QBAND(W, m, i, interp_i, ninterps, n);
        int imax = (m < i) ? m : i;
        for (int s = 1; s <= imax; ++s)
            QRHS(B, i - s, interp_i, ninterps) -=
                QBAND(W, m - s, i, interp_i, ninterps, n) * QRHS(B, i, interp_i, ninterps);
    }
}

// Differentiate the solved B-spline at left breakpoint x_i to fill c1..c5 for
// segment i of spline interp_i. coef = solved B-spline coefficients (= B).
CUDA_DEVICE
void extract_quintic_coeffs(int i, int length, int interp_i, int ninterps,
                            double *coef, double *x,
                            double *c1, double *c2, double *c3, double *c4, double *c5)
{
    double *xs = &x[interp_i * length];

    int l = quintic_find_span(xs, length, xs[i]);
    double ders[QUINTIC_DEG + 1][QUINTIC_DEG + 1];
    quintic_ders_basis_funs(xs, length, l, xs[i], ders);

    double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0;
    for (int jj = 0; jj <= QUINTIC_DEG; ++jj)
    {
        // coef (= solved B, interp_i-innermost). Outputs c1..c5 below keep their
        // interp_i*length+i layout.
        double cc = QRHS(coef, l - QUINTIC_DEG + jj, interp_i, ninterps);
        s1 += cc * ders[1][jj];
        s2 += cc * ders[2][jj];
        s3 += cc * ders[3][jj];
        s4 += cc * ders[4][jj];
        s5 += cc * ders[5][jj];
    }
    int idx = interp_i * length + i;
    c1[idx] = s1;
    c2[idx] = s2 / 2.0;
    c3[idx] = s3 / 6.0;
    c4[idx] = s4 / 24.0;
    c5[idx] = s5 / 120.0;
}

// ===========================================================================
// SPIKE / chunked parallel banded-solve primitives (see
// docs/superpowers/specs/2026-06-18-quintic-gpu-spike-solve-design.md).
// These operate on a CONTIGUOUS local band buffer ws[band_row*n + col] with a
// runtime half-bandwidth m, so the same code serves the chunk solve (m=4) and
// the reduced system (m=8). Deliberately independent of the global QBAND/QRHS
// layout: a chunk is gathered into a contiguous buffer first (band_gather).
// ===========================================================================

// Pivot-free banded LU on a contiguous buffer, half-band m, order n. In place.
// Returns 0 on success, (row+1) on a zero pivot.
CUDA_DEVICE
int banfac_local(double *ws, int m, int n)
{
    for (int i = 0; i < n; ++i)
    {
        double piv = ws[m * n + i];
        if (piv == 0.0) return i + 1;
        int imax = (m < (n - 1 - i)) ? m : (n - 1 - i);
        for (int s = 1; s <= imax; ++s)
        {
            double fac = ws[(m + s) * n + i] / piv;
            ws[(m + s) * n + i] = fac;
            for (int r = 1; r <= imax; ++r)
                ws[(m + s - r) * n + (i + r)] -= fac * ws[(m - r) * n + (i + r)];
        }
    }
    return 0;
}

// Solve after banfac_local; rhs b (length n) overwritten with the solution.
CUDA_DEVICE
void banslv_local(double *ws, int m, int n, double *b)
{
    for (int i = 0; i < n; ++i)                 // forward (unit lower L)
    {
        int imax = (m < (n - 1 - i)) ? m : (n - 1 - i);
        for (int s = 1; s <= imax; ++s)
            b[i + s] -= ws[(m + s) * n + i] * b[i];
    }
    for (int i = n - 1; i >= 0; --i)            // back (upper U)
    {
        b[i] /= ws[m * n + i];
        int imax = (m < i) ? m : i;
        for (int s = 1; s <= imax; ++s)
            b[i - s] -= ws[(m - s) * n + i] * b[i];
    }
}

// Gather band rows of columns [r0, r0+rows) of spline interp_i from the global
// (QBAND-layout) band W into a contiguous buffer dst[band_row*rows + (col-r0)].
// Copies all QUINTIC_BAND_ROWS rows; entries referencing columns outside the
// chunk are the inter-chunk coupling, handled by quintic_spike_solve (Task 2).
CUDA_DEVICE
void band_gather(double *dst, double *W, int interp_i, int ninterps,
                 int length, int r0, int rows)
{
    for (int br = 0; br < QUINTIC_BAND_ROWS; ++br)
        for (int c = 0; c < rows; ++c)
            dst[br * rows + c] = QBAND(W, br, r0 + c, interp_i, ninterps, length);
}

// Add v to reduced-system band entry A_red[i,k] (contiguous, half-band Mred).
static CUDA_DEVICE void red_add(double *wr, int Mred, int R, int i, int k, double v)
{
    wr[(size_t)(Mred + (i - k)) * R + k] += v;
}

// ===========================================================================
// Unified chunked SPIKE solve shared by the CPU mirror (serial loops) and the
// GPU kernels (block per (system, chunk), chunk band in shared memory). The
// per-chunk math lives in three CUDA_DEVICE helpers compiled by BOTH g++ and
// nvcc; only the driver differs -- quintic_spike_batched (CPU, this file) walks
// (system, chunk) serially with a heap chunk buffer, while quintic_spike_solve_gpu
// (the __CUDACC__ block) launches the same helpers as kernels with the chunk band
// in dynamic __shared__ memory. This is the "CPU mirrors GPU exactly"
// verification vehicle: because the CPU runs the identical helpers + recursion,
// the scipy parity tests validate the parallel algorithm itself.
//
// Per-system scratch (contiguous, packed by GLOBAL row -- no per-chunk Cmax
// over-allocation):
//   g[row]        local solutions g_j = D_j^{-1} f_j          (length n)
//   V[b*n + row]  right spikes, column-contiguous              (m*n)
//   W[b*n + row]  left  spikes, column-contiguous              (m*n)
//   wr / rr       reduced band (half-band Mred=3m) / rhs       ((2Mred+1)R / R)
// Interleaved reduced positions (closed form): pos_t(j)=(2j-1)m (j>=1);
// pos_b(j)=(j==0)?0:2jm (j<=P-2); R = 2(P-1)m.
// ===========================================================================
static CUDA_DEVICE int qsp_pos_t(int j, int m) { return (2 * j - 1) * m; }
static CUDA_DEVICE int qsp_pos_b(int j, int m) { return (j == 0) ? 0 : (2 * j * m); }

// Chunk size (host, both builds). Defaults to QSPIKE_DEFAULT_C, clamped to >= 2m
// (the spike structure needs >= 2m rows/chunk). Independent of length -- the
// "fixed-size buffers for long signals" property. C is NOT shrunk to fit shared
// memory as m grows down the recursion (that would collapse the reduced-system
// shrink ratio); instead the GPU factor kernel decides shared-vs-global per launch
// from whether (2m+1)*nj_max fits the smem budget.
#define QSPIKE_DEFAULT_C 1024
static inline int qspike_chunk_size(int m, int req)
{
    int C = (req > 0) ? req : QSPIKE_DEFAULT_C;
    if (C < 2 * m) C = 2 * m;
    return C;
}

// Number of chunks for order n, chunk size C, half-band m. Ceil chunking; if the
// tail chunk would be shorter than 2m (too small for both interface tips) it is
// merged into the previous chunk. Guarantees every chunk has >= 2m rows and the
// largest chunk has < C + 2m rows (bounds the shared-memory band). Chunk j spans
// rows [j*C, (j<P-1)?(j+1)*C : n).
static inline int qspike_num_chunks(int n, int C, int m)
{
    int P = (n + C - 1) / C;
    if (P < 1) P = 1;
    if (P > 1 && (n - (P - 1) * C) < 2 * m) --P;   // merge short tail
    return P;
}

// --- Per-chunk device helpers (shared by the CPU mirror and the GPU kernels) ---

// Factor chunk j (rows [s, s+nj)) of contiguous band cb[br*n+col] (half-band m,
// order n) into the caller's band buffer sD[br*nj+col] (dynamic __shared__ on the
// GPU, heap on the CPU). Solves the local rhs g[s..) = D_j^{-1} f_j and the chunk's
// right/left interface spikes into V/W (column-contiguous, packed by global row).
static CUDA_DEVICE
void qspike_factor_chunk(const double *cb, int n, int m, int s, int nj, int j, int P,
                         double *sD, const double *rb, double *g, double *V, double *W)
{
    int e = s + nj;
    for (int br = 0; br <= 2 * m; ++br)                 // copy chunk band; zero coupling rows
        for (int c = 0; c < nj; ++c)
        {
            int row = (s + c) + (br - m);
            sD[br * nj + c] = (row >= s && row < e) ? cb[(size_t)br * n + (s + c)] : 0.0;
        }
    banfac_local(sD, m, nj);

    for (int c = 0; c < nj; ++c) g[s + c] = rb[s + c];  // local solution g_j
    banslv_local(sD, m, nj, &g[s]);

    if (j < P - 1)                                      // right spike: bottom m rows = Sup_j
        for (int b = 0; b < m; ++b)
        {
            double *col = &V[(size_t)b * n + s];        // column-contiguous -> solve in place
            for (int c = 0; c < nj; ++c) col[c] = 0.0;
            for (int a = b; a < m; ++a) col[nj - m + a] = cb[(size_t)(a - b) * n + (e + b)];
            banslv_local(sD, m, nj, col);
        }
    if (j > 0)                                          // left spike: top m rows = Sub_j
        for (int b = 0; b < m; ++b)
        {
            double *col = &W[(size_t)b * n + s];
            for (int c = 0; c < nj; ++c) col[c] = 0.0;
            for (int a = 0; a <= b; ++a) col[a] = cb[(size_t)(2 * m + a - b) * n + (s - m + b)];
            banslv_local(sD, m, nj, col);
        }
}

// Assemble chunk j's rows of the reduced band wr (half-band Mred, order R) + rhs rr
// from its own spike tips. Each chunk owns reduced rows pos_t(j)/pos_b(j) -> the
// chunks write disjoint rows (no races). wr must be pre-zeroed.
static CUDA_DEVICE
void qspike_assemble_reduced_chunk(const double *g, const double *V, const double *W,
                                   double *wr, double *rr, int n, int m, int R, int Mred,
                                   int j, int P, int s, int nj)
{
    if (j >= 1)                                         // top equation, p = pos_t(j)
    {
        int p = qsp_pos_t(j, m);
        for (int a = 0; a < m; ++a) { red_add(wr, Mred, R, p + a, p + a, 1.0); rr[p + a] = g[s + a]; }
        if (j < P - 1)
        {
            int cp = qsp_pos_t(j + 1, m);
            for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b)
                red_add(wr, Mred, R, p + a, cp + b, V[(size_t)b * n + (s + a)]);
        }
        int cw = qsp_pos_b(j - 1, m);
        for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b)
            red_add(wr, Mred, R, p + a, cw + b, W[(size_t)b * n + (s + a)]);
    }
    if (j < P - 1)                                      // bottom equation, p = pos_b(j)
    {
        int p = qsp_pos_b(j, m);
        for (int a = 0; a < m; ++a) { red_add(wr, Mred, R, p + a, p + a, 1.0); rr[p + a] = g[s + nj - m + a]; }
        int cv = qsp_pos_t(j + 1, m);
        for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b)
            red_add(wr, Mred, R, p + a, cv + b, V[(size_t)b * n + (s + nj - m + a)]);
        if (j >= 1)
        {
            int cw = qsp_pos_b(j - 1, m);
            for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b)
                red_add(wr, Mred, R, p + a, cw + b, W[(size_t)b * n + (s + nj - m + a)]);
        }
    }
}

// Back-substitute chunk j: x_j = g_j - V_j x_{j+1}^t - W_j x_{j-1}^b, written into
// rb (the system rhs, overwritten with the solution), using the interface unknowns
// rr from the reduced solve. No band access -> no shared memory needed here.
static CUDA_DEVICE
void qspike_backsub_chunk(double *rb, int n, int m, const double *g,
                          const double *V, const double *W, const double *rr,
                          int j, int P, int s, int nj)
{
    for (int c = 0; c < nj; ++c)
    {
        double x = g[s + c];
        if (j < P - 1) { int cp = qsp_pos_t(j + 1, m); for (int b = 0; b < m; ++b) x -= V[(size_t)b * n + (s + c)] * rr[cp + b]; }
        if (j > 0)     { int cp = qsp_pos_b(j - 1, m); for (int b = 0; b < m; ++b) x -= W[(size_t)b * n + (s + c)] * rr[cp + b]; }
        rb[s + c] = x;
    }
}

#ifndef __CUDACC__
// ===========================================================================
// CPU mirror of the GPU SPIKE launcher (host only -- the GPU build uses
// quintic_spike_solve_gpu in the __CUDACC__ block). Solves nsys independent
// banded systems (band cb[sys*(2m+1)*n + br*n + col], rhs rb[sys*n + col],
// overwritten with the solution) by the chunked SPIKE method, recursing on the
// reduced system. Runs the SAME per-chunk device helpers as the GPU kernels,
// serially. nsys is constant across recursion levels (each system spawns exactly
// one reduced system), so the recursion is a clean batched re-entry.
// ===========================================================================
void quintic_spike_batched(double *cb, double *rb, int nsys, int n, int m, int Creq)
{
    int C = qspike_chunk_size(m, Creq);
    int P = qspike_num_chunks(n, C, m);
    int bandrows = 2 * m + 1;

    if (P == 1)                                         // single chunk: direct banded solve
    {
        for (int sys = 0; sys < nsys; ++sys)
        {
            double *A = &cb[(size_t)sys * bandrows * n];
            banfac_local(A, m, n);
            banslv_local(A, m, n, &rb[(size_t)sys * n]);
        }
        return;
    }

    int Mred = 3 * m;
    int R = 2 * (P - 1) * m;
    int last = n - (P - 1) * C;                         // size of the final (largest) chunk
    int nj_max = (last > C) ? last : C;                 // < C + 2m -> bounds the chunk band

    double *g  = new double[(size_t)nsys * n];
    double *V  = new double[(size_t)nsys * m * n];
    double *W  = new double[(size_t)nsys * m * n];
    double *wr = new double[(size_t)nsys * (2 * Mred + 1) * R]();   // zero-initialized
    double *rr = new double[(size_t)nsys * R];
    double *sD = new double[(size_t)bandrows * nj_max];            // chunk band scratch (mirrors GPU __shared__)

    for (int sys = 0; sys < nsys; ++sys)                // phase 1: factor + assemble reduced
    {
        double *cbs = &cb[(size_t)sys * bandrows * n];
        double *rbs = &rb[(size_t)sys * n];
        double *gs  = &g[(size_t)sys * n];
        double *Vs  = &V[(size_t)sys * m * n];
        double *Ws  = &W[(size_t)sys * m * n];
        double *wrs = &wr[(size_t)sys * (2 * Mred + 1) * R];
        double *rrs = &rr[(size_t)sys * R];
        for (int j = 0; j < P; ++j)
        {
            int s = j * C;
            int e = (j < P - 1) ? (s + C) : n;
            int nj = e - s;
            qspike_factor_chunk(cbs, n, m, s, nj, j, P, sD, rbs, gs, Vs, Ws);
            qspike_assemble_reduced_chunk(gs, Vs, Ws, wrs, rrs, n, m, R, Mred, j, P, s, nj);
        }
    }

    if (R > C && 2 * R <= n)                            // phase 2: reduced solve (recurse if it shrinks)
        quintic_spike_batched(wr, rr, nsys, R, Mred, Creq);
    else
        for (int sys = 0; sys < nsys; ++sys)
        {
            banfac_local(&wr[(size_t)sys * (2 * Mred + 1) * R], Mred, R);
            banslv_local(&wr[(size_t)sys * (2 * Mred + 1) * R], Mred, R, &rr[(size_t)sys * R]);
        }

    for (int sys = 0; sys < nsys; ++sys)                // phase 3: back-substitute
    {
        double *rbs = &rb[(size_t)sys * n];
        double *gs  = &g[(size_t)sys * n];
        double *Vs  = &V[(size_t)sys * m * n];
        double *Ws  = &W[(size_t)sys * m * n];
        double *rrs = &rr[(size_t)sys * R];
        for (int j = 0; j < P; ++j)
        {
            int s = j * C;
            int e = (j < P - 1) ? (s + C) : n;
            int nj = e - s;
            qspike_backsub_chunk(rbs, n, m, gs, Vs, Ws, rrs, j, P, s, nj);
        }
    }

    delete[] g; delete[] V; delete[] W; delete[] wr; delete[] rr; delete[] sD;
}

// Solve all splines: gather each spline's band (QBAND W) + rhs (QRHS B) into a
// contiguous batched buffer, run the SPIKE mirror, scatter the solution back into
// B. chunk<=0 keeps the legacy CPU behaviour (single direct banded solve per
// spline); a positive chunk forces the chunked/recursive path (exercised by the
// tests). The GPU path defaults chunk to QSPIKE_DEFAULT_C to saturate the device.
void quintic_spike_solve(double *W, double *B, int ninterps, int length, int chunk, int uniform)
{
    (void)uniform;   // uniform Toeplitz caching is a follow-up; not on the mirrored path
    int m = QUINTIC_HALF_BAND;
    int bandrows = QUINTIC_BAND_ROWS;
    int Creq = (chunk > 0) ? chunk : length;            // CPU default: one chunk (P==1)

    double *cb = new double[(size_t)ninterps * bandrows * length];
    double *rb = new double[(size_t)ninterps * length];
    for (int s = 0; s < ninterps; ++s)
    {
        band_gather(&cb[(size_t)s * bandrows * length], W, s, ninterps, length, 0, length);
        double *rbs = &rb[(size_t)s * length];
        for (int c = 0; c < length; ++c) rbs[c] = QRHS(B, c, s, ninterps);
    }
    quintic_spike_batched(cb, rb, ninterps, length, m, Creq);
    for (int s = 0; s < ninterps; ++s)
        for (int c = 0; c < length; ++c) QRHS(B, c, s, ninterps) = rb[(size_t)s * length + c];
    delete[] cb; delete[] rb;
}
#endif // !__CUDACC__

// --- Kernels (block per spline over rows; mirror fill_B / set_spline_constants) ---
CUDA_KERNEL
void fill_quintic_band(double *x, double *y, double *W, double *B,
                       int ninterps, int length)
{
#ifdef __CUDACC__
    int start1 = blockIdx.x;
    int end1 = ninterps;
    int diff1 = gridDim.x;
#else
    int start1 = 0;
    int end1 = ninterps;
    int diff1 = 1;
#endif
    for (int interp_i = start1; interp_i < end1; interp_i += diff1)
    {
#ifdef __CUDACC__
        int start2 = threadIdx.x;
        int end2 = length;
        int diff2 = blockDim.x;
#else
        int start2 = 0;
        int end2 = length;
        int diff2 = 1;
#endif
        for (int i = start2; i < end2; i += diff2)
            prep_quintic_band(i, length, interp_i, ninterps, W, B, x, y);
    }
}

CUDA_KERNEL
void solve_quintic_band_batch(double *W, double *B, int ninterps, int length)
{
#ifdef __CUDACC__
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int diff = gridDim.x * blockDim.x;
#else
    int start = 0;
    int diff = 1;
#endif
    for (int interp_i = start; interp_i < ninterps; interp_i += diff)
    {
        // interp_i-innermost layout -> consecutive threads (interp_i, interp_i+1,
        // ...) read consecutive addresses: the memory-bound solve coalesces.
        int info = quintic_banfac(W, interp_i, ninterps, length);
        if (info == 0) quintic_banslv(W, B, interp_i, ninterps, length);
    }
}

CUDA_KERNEL
void set_quintic_constants(double *x, double *coef,
                           double *c1, double *c2, double *c3, double *c4, double *c5,
                           int ninterps, int length)
{
#ifdef __CUDACC__
    int start1 = blockIdx.x;
    int end1 = ninterps;
    int diff1 = gridDim.x;
#else
    int start1 = 0;
    int end1 = ninterps;
    int diff1 = 1;
#endif
    for (int interp_i = start1; interp_i < end1; interp_i += diff1)
    {
#ifdef __CUDACC__
        int start2 = threadIdx.x;
        int end2 = length - 1;
        int diff2 = blockDim.x;
#else
        int start2 = 0;
        int end2 = length - 1;
        int diff2 = 1;
#endif
        for (int i = start2; i < end2; i += diff2)
            extract_quintic_coeffs(i, length, interp_i, ninterps, coef, x, c1, c2, c3, c4, c5);
    }
}

#ifdef __CUDACC__
// ===========================================================================
// GPU SPIKE solve: host-recursive launcher that MIRRORS quintic_spike_batched
// (the CPU driver in the #ifndef block) phase-for-phase, calling the SAME
// per-chunk CUDA_DEVICE helpers (qspike_factor_chunk / _assemble_reduced_chunk /
// _backsub_chunk + banfac_local/banslv_local). The only difference vs the CPU
// path is parallelism: a block per (system, chunk) for phases 1 & 3, a thread
// per system for the direct (single-chunk / base-case reduced) solves, and the
// reduced system solved by the same host launcher re-entered recursively.
//
// !!! UNVERIFIED: this machine has no nvcc/GPU, so the device code below has
// never been compiled or run. It is a faithful mirror of the CPU reference
// (which IS verified vs scipy). Build + run tests on a CUDA box before trusting.
//
// Per-system scratch (contiguous, packed by GLOBAL row -- identical layout to the
// CPU mirror; no per-chunk Cmax over-allocation):
//   cb[sys*bandrows*n + br*n + col]  contiguous band (half-band m, order n)
//   rb[sys*n + col]                  rhs, overwritten with the solution
//   g [sys*n + row]                  local solutions g_j
//   V [sys*m*n + b*n + row]          right spikes, column-contiguous
//   W [sys*m*n + b*n + row]          left  spikes, column-contiguous
//   wr[sys*(2Mred+1)*R + ...]        reduced band (half-band Mred=3m, order R)
//   rr[sys*R + ...]                  reduced rhs
// The reduced band wr (per-system stride (2Mred+1)*R) is EXACTLY the cb of the
// recursive call (bandrows'=2Mred+1, n'=R, m'=Mred), rr its rb -- so the recursion
// is a clean batched re-entry with nsys constant across levels.
//
// qsp_pos_t / qsp_pos_b are shared (defined ONCE outside this guard).
// ===========================================================================

#define QSPIKE_SMEM_BYTES (96 * 1024)   // per-block dynamic-smem ceiling for the chunk band

// Direct banded solve, one thread per system: banfac_local + banslv_local on the
// full system. Used for P==1 and for the base case of the reduced solve.
CUDA_KERNEL
void qspike_direct_solve_kernel(double *cb, double *rb, int nsys, int n, int m)
{
    int bandrows = 2 * m + 1;
    int sys = blockIdx.x * blockDim.x + threadIdx.x;
    if (sys >= nsys) return;
    double *A = &cb[(size_t)sys * bandrows * n];
    banfac_local(A, m, n);
    banslv_local(A, m, n, &rb[(size_t)sys * n]);
}

// Phase 1 (block per (system, chunk)): factor chunk j of system sys and assemble
// its disjoint rows of the reduced system. grid = dim3(P, nsys): chunk j on x,
// system sys on y. The chunk band sD lives in dynamic __shared__ when it fits the
// smem budget (sDg == nullptr), else in a global per-block slice of sDg indexed by
// (blockIdx.y*P + blockIdx.x). Either way the band buffer is private to the block,
// so the two storage choices give bit-identical results. The serial banded factor
// runs under THREAD_ZERO (v1: one thread does the chunk; other threads idle). All
// (sys,chunk) blocks write DISJOINT g/V/W ranges and DISJOINT reduced rows (chunk j
// owns reduced rows pos_t(j)/pos_b(j)), so there are no cross-block races; wr is
// pre-zeroed by cudaMemset before launch.
CUDA_KERNEL
void qspike_factor_kernel(const double *cb, const double *rb, double *g, double *V,
                          double *W, double *wr, double *rr, double *sDg,
                          int nsys, int n, int m, int C, int P, int Mred, int R,
                          int nj_max)
{
    extern __shared__ double smem[];
    int j = blockIdx.x;
    int sys = blockIdx.y;
    if (j >= P || sys >= nsys) return;
    int bandrows = 2 * m + 1;
    int s = j * C;
    int e = (j < P - 1) ? (s + C) : n;
    int nj = e - s;

    // chunk band scratch: dynamic shared (preferred) or a per-block global slice.
    double *sD = (sDg != nullptr)
                     ? &sDg[((size_t)blockIdx.y * P + blockIdx.x) * bandrows * nj_max]
                     : smem;

    if (THREAD_ZERO)
    {
        const double *cbs = &cb[(size_t)sys * bandrows * n];
        const double *rbs = &rb[(size_t)sys * n];
        double *gs  = &g[(size_t)sys * n];
        double *Vs  = &V[(size_t)sys * m * n];
        double *Ws  = &W[(size_t)sys * m * n];
        double *wrs = &wr[(size_t)sys * (2 * Mred + 1) * R];
        double *rrs = &rr[(size_t)sys * R];
        qspike_factor_chunk(cbs, n, m, s, nj, j, P, sD, rbs, gs, Vs, Ws);
        qspike_assemble_reduced_chunk(gs, Vs, Ws, wrs, rrs, n, m, R, Mred, j, P, s, nj);
    }
}

// Phase 3 (block per (system, chunk)): back-substitute chunk j of system sys using
// the reduced solution rr. No band access -> no shared memory. Disjoint writes
// (each chunk owns rows [s, s+nj) of rb).
CUDA_KERNEL
void qspike_backsub_kernel(double *rb, const double *g, const double *V,
                           const double *W, const double *rr,
                           int nsys, int n, int m, int C, int P, int R)
{
    int j = blockIdx.x;
    int sys = blockIdx.y;
    if (j >= P || sys >= nsys) return;
    if (!THREAD_ZERO) return;
    int s = j * C;
    int e = (j < P - 1) ? (s + C) : n;
    int nj = e - s;
    qspike_backsub_chunk(&rb[(size_t)sys * n], n, m, &g[(size_t)sys * n],
                         &V[(size_t)sys * m * n], &W[(size_t)sys * m * n],
                         &rr[(size_t)sys * R], j, P, s, nj);
}

// Host-recursive batched SPIKE launcher (GPU). MIRRORS quintic_spike_batched: same
// chunking, same phases, same recurse-iff-shrinks decision. Host code only does
// cudaMalloc/memset/launch/sync/free and the R>C decision -- every CUDA_DEVICE call
// happens inside a kernel. cb/rb are device pointers (per-system contiguous band /
// rhs); rb is overwritten with the solution.
void qspike_solve_gpu_batched(double *cb, double *rb, int nsys, int n, int m, int Creq)
{
    int C = qspike_chunk_size(m, Creq);
    int P = qspike_num_chunks(n, C, m);
    int bandrows = 2 * m + 1;
    int T = NUM_THREADS_INTERPOLATE;

    if (P == 1)                                         // single chunk: direct banded solve
    {
        qspike_direct_solve_kernel<<<(nsys + T - 1) / T, T>>>(cb, rb, nsys, n, m);
        cudaDeviceSynchronize(); gpuErrchk(cudaGetLastError());
        return;
    }

    int Mred = 3 * m;
    int R = 2 * (P - 1) * m;
    int last = n - (P - 1) * C;                         // size of the final (largest) chunk
    int nj_max = (last > C) ? last : C;                 // < C + 2m -> bounds the chunk band

    double *g, *V, *W, *wr, *rr;
    gpuErrchk(cudaMalloc(&g,  (size_t)nsys * n * sizeof(double)));
    gpuErrchk(cudaMalloc(&V,  (size_t)nsys * m * n * sizeof(double)));
    gpuErrchk(cudaMalloc(&W,  (size_t)nsys * m * n * sizeof(double)));
    gpuErrchk(cudaMalloc(&wr, (size_t)nsys * (2 * Mred + 1) * R * sizeof(double)));
    gpuErrchk(cudaMalloc(&rr, (size_t)nsys * R * sizeof(double)));
    gpuErrchk(cudaMemset(wr, 0, (size_t)nsys * (2 * Mred + 1) * R * sizeof(double)));   // reduced band pre-zeroed

    // --- phase 1: factor + assemble reduced, block per (system, chunk) ---
    // Shared-vs-global chunk band decision: prefer dynamic __shared__ when the
    // largest chunk band (bandrows*nj_max doubles) fits the per-block ceiling. The
    // long-spline production path (large C at the shallow levels) uses shared mem;
    // only levels whose chunk band exceeds the ceiling fall back to a global slice
    // (identical results, just slower memory). The decision is per launch (per
    // recursion level), since m grows (Mred=3m) and C shrinks toward the base case.
    size_t smem = (size_t)bandrows * nj_max * sizeof(double);
    dim3 grid(P, nsys);
    double *sDg = nullptr;
    if (smem <= (size_t)QSPIKE_SMEM_BYTES)
    {
        if (smem > 48 * 1024)
            gpuErrchk(cudaFuncSetAttribute(qspike_factor_kernel,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           (int)smem));
        qspike_factor_kernel<<<grid, T, smem>>>(cb, rb, g, V, W, wr, rr, nullptr,
                                                nsys, n, m, C, P, Mred, R, nj_max);
    }
    else                                                // chunk band too big for smem -> global per-block slice
    {
        gpuErrchk(cudaMalloc(&sDg, (size_t)P * nsys * bandrows * nj_max * sizeof(double)));
        qspike_factor_kernel<<<grid, T>>>(cb, rb, g, V, W, wr, rr, sDg,
                                          nsys, n, m, C, P, Mred, R, nj_max);
    }
    cudaDeviceSynchronize(); gpuErrchk(cudaGetLastError());
    if (sDg) gpuErrchk(cudaFree(sDg));

    // --- phase 2: reduced solve (recurse iff it shrinks, else direct base case) ---
    if (R > C && 2 * R <= n)
        qspike_solve_gpu_batched(wr, rr, nsys, R, Mred, Creq);
    else
    {
        qspike_direct_solve_kernel<<<(nsys + T - 1) / T, T>>>(wr, rr, nsys, R, Mred);
        cudaDeviceSynchronize(); gpuErrchk(cudaGetLastError());
    }

    // --- phase 3: back-substitute, block per (system, chunk) ---
    qspike_backsub_kernel<<<grid, T>>>(rb, g, V, W, rr, nsys, n, m, C, P, R);
    cudaDeviceSynchronize(); gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(g)); gpuErrchk(cudaFree(V)); gpuErrchk(cudaFree(W));
    gpuErrchk(cudaFree(wr)); gpuErrchk(cudaFree(rr));
}

// Gather each spline's QBAND band + QRHS rhs into the contiguous batched buffers
// cb0/rb0 (the layout qspike_solve_gpu_batched expects). grid = dim3(systems-on-x,
// columns-on-threads); one block per system, threads stride over columns so the
// QBAND reads coalesce across threads. Each thread copies all QUINTIC_BAND_ROWS
// rows of its column plus the rhs.
CUDA_KERNEL
void qspike_gather_kernel(const double *W, const double *B, double *cb0, double *rb0,
                          int ninterps, int length)
{
    int bandrows = QUINTIC_BAND_ROWS;
    for (int sys = BLOCK_START_X; sys < ninterps; sys += GRID_INCR_X)
    {
        double *cbs = &cb0[(size_t)sys * bandrows * length];
        double *rbs = &rb0[(size_t)sys * length];
        for (int c = THREAD_START_X; c < length; c += BLOCK_INCR_X)
        {
            for (int br = 0; br < bandrows; ++br)
                cbs[(size_t)br * length + c] = QBAND(W, br, c, sys, ninterps, length);
            rbs[c] = QRHS(B, c, sys, ninterps);
        }
    }
}

// Scatter the solved rhs rb0 back into the global QRHS layout B.
CUDA_KERNEL
void qspike_scatter_kernel(double *B, const double *rb0, int ninterps, int length)
{
    for (int sys = BLOCK_START_X; sys < ninterps; sys += GRID_INCR_X)
    {
        const double *rbs = &rb0[(size_t)sys * length];
        for (int c = THREAD_START_X; c < length; c += BLOCK_INCR_X)
            QRHS(B, c, sys, ninterps) = rbs[c];
    }
}

// Host entry called by interpolate_quintic. Gathers QBAND/QRHS -> contiguous
// batched cb0/rb0, runs the recursive batched SPIKE solver, scatters back.
void quintic_spike_solve_gpu(double *W, double *B, int ninterps, int length, int chunk, int uniform)
{
    (void)uniform;                                  // Toeplitz cache is a GPU follow-up
    int m = QUINTIC_HALF_BAND;
    int bandrows = QUINTIC_BAND_ROWS;
    int Creq = (chunk > 0) ? chunk : QSPIKE_DEFAULT_C;   // GPU default: 1024 (saturate the device)
    int T = NUM_THREADS_INTERPOLATE;

    double *cb0, *rb0;
    gpuErrchk(cudaMalloc(&cb0, (size_t)ninterps * bandrows * length * sizeof(double)));
    gpuErrchk(cudaMalloc(&rb0, (size_t)ninterps * length * sizeof(double)));

    int gblocks = (ninterps < 65535) ? ninterps : 65535;
    qspike_gather_kernel<<<gblocks, T>>>(W, B, cb0, rb0, ninterps, length);
    cudaDeviceSynchronize(); gpuErrchk(cudaGetLastError());

    qspike_solve_gpu_batched(cb0, rb0, ninterps, length, m, Creq);

    qspike_scatter_kernel<<<gblocks, T>>>(B, rb0, ninterps, length);
    cudaDeviceSynchronize(); gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(cb0)); gpuErrchk(cudaFree(rb0));
}
#endif // __CUDACC__

// Host orchestrator. Internally allocates the band scratch W (zeroed) and the
// B-spline coefficient scratch B; fills c1..c5 in place.
void interpolate_quintic(double *x, double *y,
                         double *c1, double *c2, double *c3, double *c4, double *c5,
                         int length, int ninterps, int chunk, int uniform)
{
    size_t band_count = (size_t)ninterps * QUINTIC_BAND_ROWS * (size_t)length;
    size_t rhs_count = (size_t)ninterps * (size_t)length;

    // Default solve: SPIKE chunked/parallel banded solve. Define
    // GBT_QUINTIC_LEGACY_SOLVE to fall back to the one-thread-per-spline solve
    // (e.g. if the GPU SPIKE path needs validating against the legacy baseline).
#ifdef __CUDACC__
    double *W;
    double *B;
    gpuErrchk(cudaMalloc(&W, band_count * sizeof(double)));
    gpuErrchk(cudaMalloc(&B, rhs_count * sizeof(double)));
    gpuErrchk(cudaMemset(W, 0, band_count * sizeof(double)));

    fill_quintic_band<<<ninterps, NUM_THREADS_INTERPOLATE>>>(x, y, W, B, ninterps, length);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

#ifdef GBT_QUINTIC_LEGACY_SOLVE
    (void)chunk; (void)uniform;
    int sblocks = std::ceil((ninterps + NUM_THREADS_INTERPOLATE - 1) / NUM_THREADS_INTERPOLATE);
    solve_quintic_band_batch<<<sblocks, NUM_THREADS_INTERPOLATE>>>(W, B, ninterps, length);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    quintic_spike_solve_gpu(W, B, ninterps, length, chunk, uniform);   // GPU SPIKE (UNVERIFIED)
#endif

    set_quintic_constants<<<ninterps, NUM_THREADS_INTERPOLATE>>>(x, B, c1, c2, c3, c4, c5, ninterps, length);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(W));
    gpuErrchk(cudaFree(B));
#else
    double *W = new double[band_count]();   // value-initialized to 0
    double *B = new double[rhs_count];

    fill_quintic_band(x, y, W, B, ninterps, length);
#ifdef GBT_QUINTIC_LEGACY_SOLVE
    (void)chunk; (void)uniform;
    solve_quintic_band_batch(W, B, ninterps, length);
#else
    quintic_spike_solve(W, B, ninterps, length, chunk, uniform);   // CPU SPIKE (verified)
#endif
    set_quintic_constants(x, B, c1, c2, c3, c4, c5, ninterps, length);

    delete[] W;
    delete[] B;
#endif
}

CUDA_KERNEL
void eval_quintic_kernel(QuinticSpline *spline, double *y_new, double *x_new, int *spline_index, int N)
{
    spline->eval(y_new, x_new, spline_index, N);
}

void eval_quintic_wrap(QuinticSpline *spline, double *y_new, double *x_new, int *spline_index, int N)
{
#ifdef __CUDACC__
    int nblocks = std::ceil((N + NUM_THREADS_INTERPOLATE - 1) / NUM_THREADS_INTERPOLATE);

    QuinticSpline *d_spline;
    gpuErrchk(cudaMalloc(&d_spline, sizeof(QuinticSpline)));
    gpuErrchk(cudaMemcpy(d_spline, spline, sizeof(QuinticSpline), cudaMemcpyHostToDevice));

    eval_quintic_kernel<<<nblocks, NUM_THREADS_INTERPOLATE>>>(d_spline, y_new, x_new, spline_index, N);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(d_spline));
#else
    spline->eval(y_new, x_new, spline_index, N);
#endif
}

// CubicSpline::even_sampled_search / binary_search moved to
// InterpolateDevice.hh (header-only).



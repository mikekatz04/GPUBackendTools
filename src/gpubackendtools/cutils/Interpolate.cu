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
static inline void red_add(double *wr, int Mred, int R, int i, int k, double v)
{
    wr[(size_t)(Mred + (i - k)) * R + k] += v;
}

static inline bool darr_eq(const double *a, const double *b, size_t k)
{ for (size_t i = 0; i < k; ++i) if (a[i] != b[i]) return false; return true; }
static inline void darr_cpy(double *d, const double *s, size_t k)
{ for (size_t i = 0; i < k; ++i) d[i] = s[i]; }

// Solve ONE banded system: contiguous band cb[br*n + col] (half-band m), rhs rb
// (length n) overwritten with the solution. SPIKE partition into chunks of C
// rows (>= 2m); reduced system assembled in interleaved physical order and
// solved directly with the same banded primitives (Task 3 adds recursion).
// Mirrors the validated NumPy reference (matches np.linalg.solve to ~1e-16).
// uniform != 0 enables the Toeplitz factorization cache for interior chunks.
void spike_solve_one(double *cb, double *rb, int n, int m, int C, int uniform)
{
    if (C < 2 * m) C = 2 * m;
    int P = n / C;
    if (P < 1) P = 1;
    if (P == 1)                                   // single chunk: direct banded solve
    {
        banfac_local(cb, m, n);
        banslv_local(cb, m, n, rb);
        return;
    }

    double *g = new double[n];                    // g_j = D_j^{-1} f_j
    double *Vs = new double[(size_t)n * m];       // right spikes, row-major [row*m + b]
    double *Ws = new double[(size_t)n * m];       // left spikes

    // Uniform fast path: cache one interior full-size chunk's factorization +
    // spikes and reuse for any later interior chunk whose band matches exactly
    // (Toeplitz). Element-wise compare => always correct (mismatches recompute,
    // so non-uniform input simply never reuses).
    int band_sz = (2 * m + 1) * C;
    double *cDraw = nullptr, *cDfac = nullptr, *cV = nullptr, *cW = nullptr, *cSup = nullptr, *cSub = nullptr;
    bool have_cache = false;
    if (uniform)
    {
        cDraw = new double[band_sz]; cDfac = new double[band_sz];
        cV = new double[(size_t)C * m]; cW = new double[(size_t)C * m];
        cSup = new double[m * m]; cSub = new double[m * m];
    }

    for (int j = 0; j < P; ++j)
    {
        int s = j * C;
        int e = (j < P - 1) ? (s + C) : n;
        int nj = e - s;
        bool interior = uniform && j > 0 && j < P - 1 && nj == C;

        double *D = new double[(size_t)(2 * m + 1) * nj];   // chunk diagonal block (coupling rows zeroed)
        for (int br = 0; br <= 2 * m; ++br)
            for (int c = 0; c < nj; ++c)
            {
                int row = (s + c) + (br - m);
                D[br * nj + c] = (row >= s && row < e) ? cb[br * n + (s + c)] : 0.0;
            }

        bool fill_cache = false;
        if (interior)
        {
            double *sup = new double[m * m];
            double *sub = new double[m * m];
            for (int a = 0; a < m; ++a)
                for (int b = 0; b < m; ++b)
                {
                    sup[a * m + b] = (b <= a) ? cb[(a - b) * n + (e + b)] : 0.0;
                    sub[a * m + b] = (a <= b) ? cb[(2 * m + a - b) * n + (s - m + b)] : 0.0;
                }
            if (have_cache && darr_eq(D, cDraw, band_sz) &&
                darr_eq(sup, cSup, (size_t)m * m) && darr_eq(sub, cSub, (size_t)m * m))
            {
                for (int c = 0; c < nj; ++c) g[s + c] = rb[s + c];
                banslv_local(cDfac, m, nj, &g[s]);
                for (int c = 0; c < nj; ++c)
                    for (int b = 0; b < m; ++b)
                    {
                        Vs[(size_t)(s + c) * m + b] = cV[c * m + b];
                        Ws[(size_t)(s + c) * m + b] = cW[c * m + b];
                    }
                delete[] sup; delete[] sub; delete[] D;
                continue;
            }
            darr_cpy(cDraw, D, band_sz);
            darr_cpy(cSup, sup, (size_t)m * m);
            darr_cpy(cSub, sub, (size_t)m * m);
            delete[] sup; delete[] sub;
            fill_cache = true;
        }

        banfac_local(D, m, nj);

        for (int c = 0; c < nj; ++c) g[s + c] = rb[s + c];
        banslv_local(D, m, nj, &g[s]);

        if (j < P - 1)                            // right spike: rhs bottom m rows = Sup_j (nonzero a>=b)
            for (int b = 0; b < m; ++b)
            {
                double *col = new double[nj];
                for (int c = 0; c < nj; ++c) col[c] = 0.0;
                for (int a = b; a < m; ++a) col[nj - m + a] = cb[(a - b) * n + (e + b)];
                banslv_local(D, m, nj, col);
                for (int c = 0; c < nj; ++c) Vs[(size_t)(s + c) * m + b] = col[c];
                delete[] col;
            }
        if (j > 0)                                // left spike: rhs top m rows = Sub_j (nonzero a<=b)
            for (int b = 0; b < m; ++b)
            {
                double *col = new double[nj];
                for (int c = 0; c < nj; ++c) col[c] = 0.0;
                for (int a = 0; a <= b; ++a) col[a] = cb[(2 * m + a - b) * n + (s - m + b)];
                banslv_local(D, m, nj, col);
                for (int c = 0; c < nj; ++c) Ws[(size_t)(s + c) * m + b] = col[c];
                delete[] col;
            }

        if (fill_cache)                           // remember this interior chunk for reuse
        {
            darr_cpy(cDfac, D, band_sz);
            for (int c = 0; c < C; ++c)
                for (int b = 0; b < m; ++b)
                {
                    cV[c * m + b] = Vs[(size_t)(s + c) * m + b];
                    cW[c * m + b] = Ws[(size_t)(s + c) * m + b];
                }
            have_cache = true;
        }
        delete[] D;
    }
    if (uniform) { delete[] cDraw; delete[] cDfac; delete[] cV; delete[] cW; delete[] cSup; delete[] cSub; }

    // Interleaved reduced unknowns: ("b",0),("t",1),("b",1),...,("t",P-1).
    int *pos_t = new int[P];
    int *pos_b = new int[P];
    for (int j = 0; j < P; ++j) { pos_t[j] = -1; pos_b[j] = -1; }
    int blk = 0;
    for (int j = 0; j < P; ++j)
    {
        if (j > 0) { pos_t[j] = blk * m; ++blk; }
        if (j < P - 1) { pos_b[j] = blk * m; ++blk; }
    }
    int R = blk * m;
    int Mred = 3 * m;
    double *wr = new double[(size_t)(2 * Mred + 1) * R];
    for (size_t t = 0; t < (size_t)(2 * Mred + 1) * R; ++t) wr[t] = 0.0;
    double *rr = new double[R];

    for (int j = 0; j < P; ++j)
    {
        int s = j * C;
        int e = (j < P - 1) ? (s + C) : n;
        int nj = e - s;
        if (pos_t[j] >= 0)                        // top equation: x_j^t + V_j^t x_{j+1}^t + W_j^t x_{j-1}^b = g_j^t
        {
            int p = pos_t[j];
            for (int a = 0; a < m; ++a) { red_add(wr, Mred, R, p + a, p + a, 1.0); rr[p + a] = g[s + a]; }
            if (j < P - 1 && pos_t[j + 1] >= 0)
                for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b)
                    red_add(wr, Mred, R, p + a, pos_t[j + 1] + b, Vs[(size_t)(s + a) * m + b]);
            if (j > 0 && pos_b[j - 1] >= 0)
                for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b)
                    red_add(wr, Mred, R, p + a, pos_b[j - 1] + b, Ws[(size_t)(s + a) * m + b]);
        }
        if (pos_b[j] >= 0)                        // bottom equation: x_j^b + V_j^b x_{j+1}^t + W_j^b x_{j-1}^b = g_j^b
        {
            int p = pos_b[j];
            for (int a = 0; a < m; ++a) { red_add(wr, Mred, R, p + a, p + a, 1.0); rr[p + a] = g[s + nj - m + a]; }
            if (j < P - 1 && pos_t[j + 1] >= 0)
                for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b)
                    red_add(wr, Mred, R, p + a, pos_t[j + 1] + b, Vs[(size_t)(s + nj - m + a) * m + b]);
            if (j > 0 && pos_b[j - 1] >= 0)
                for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b)
                    red_add(wr, Mred, R, p + a, pos_b[j - 1] + b, Ws[(size_t)(s + nj - m + a) * m + b]);
        }
    }

    // Solve the reduced system. It is itself banded (half-band Mred), so recurse
    // when that makes real progress (order at least halves); otherwise solve it
    // directly. (At C=2m the reduced order ~= n, so recursion would not shrink.)
    if (R > C && 2 * R <= n)
        spike_solve_one(wr, rr, R, Mred, C, 0);   // reduced system is not Toeplitz
    else
    {
        banfac_local(wr, Mred, R);
        banslv_local(wr, Mred, R, rr);            // rr now holds the interface tips
    }

    for (int j = 0; j < P; ++j)                   // x_j = g_j - V_j x_{j+1}^t - W_j x_{j-1}^b
    {
        int s = j * C;
        int e = (j < P - 1) ? (s + C) : n;
        int nj = e - s;
        for (int c = 0; c < nj; ++c)
        {
            double x = g[s + c];
            if (j < P - 1 && pos_t[j + 1] >= 0)
                for (int b = 0; b < m; ++b) x -= Vs[(size_t)(s + c) * m + b] * rr[pos_t[j + 1] + b];
            if (j > 0 && pos_b[j - 1] >= 0)
                for (int b = 0; b < m; ++b) x -= Ws[(size_t)(s + c) * m + b] * rr[pos_b[j - 1] + b];
            rb[s + c] = x;
        }
    }

    delete[] g; delete[] Vs; delete[] Ws;
    delete[] pos_t; delete[] pos_b; delete[] wr; delete[] rr;
}

// Solve all splines: gather each spline's band (QBAND W) + rhs (QRHS B) into
// contiguous buffers, SPIKE-solve, write the solution back into B. chunk<=0 =>
// single chunk (CPU auto; the GPU shared-memory budget is set in Task 5).
void quintic_spike_solve(double *W, double *B, int ninterps, int length, int chunk, int uniform)
{
    int m = QUINTIC_HALF_BAND;
    int C = (chunk > 0) ? chunk : length;
    for (int s = 0; s < ninterps; ++s)
    {
        double *cb = new double[(size_t)QUINTIC_BAND_ROWS * length];
        band_gather(cb, W, s, ninterps, length, 0, length);
        double *rb = new double[length];
        for (int c = 0; c < length; ++c) rb[c] = QRHS(B, c, s, ninterps);
        spike_solve_one(cb, rb, length, m, C, uniform);
        for (int c = 0; c < length; ++c) QRHS(B, c, s, ninterps) = rb[c];
        delete[] cb; delete[] rb;
    }
}

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
// GPU SPIKE solve: 3 kernels parallel over (spline, chunk) using GLOBAL scratch
// (no shared memory yet -- spec v1 "keep global W"; shared-memory fusion is a
// follow-up). Mirrors the CPU spike_solve_one phases. Single-level reduced solve
// (one thread per spline); GPU two-level recursion is a follow-up.
//
// !!! UNVERIFIED: this machine has no nvcc/GPU, so the device code below has
// never been compiled or run. It is a faithful mirror of the CPU reference
// (which IS verified vs scipy). Build + run tests on a CUDA box before trusting.
//
// Scratch layouts (P chunks of C rows, half-band m, reduced half-band Mred=3m,
// reduced order R = 2(P-1)m):
//   Dg : [sp][chunk] contiguous (2m+1)*C diagonal band   (factored in place)
//   gg : [sp][row]                                        (local solutions g_j)
//   Vg/Wg : [sp][chunk][b][c] contiguous in c            (right/left spikes)
//   wrg/rrg : [sp] reduced band / rhs
// Interleaved reduced positions (closed form, same for all splines):
//   pos_t(j) = (2j-1)m  (j>=1) ;  pos_b(j) = (j==0)?0:2jm  (j<=P-2)
// ===========================================================================
CUDA_DEVICE int qsp_pos_t(int j, int m) { return (2 * j - 1) * m; }
CUDA_DEVICE int qsp_pos_b(int j, int m) { return (j == 0) ? 0 : (2 * j * m); }

// Per-chunk global slices are fixed-size at Cmax = 2*C because the last chunk
// absorbs the remainder (nj in [C, 2C)). D internal row-stride stays nj; only
// the slice offsets/strides use Cmax. (The CPU reference is safe via per-nj new.)
CUDA_KERNEL
void spike_factor_kernel(double *W, double *B, double *Dg, double *gg,
                         double *Vg, double *Wg, int ninterps, int length,
                         int C, int Cmax, int P, int m)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int sp = blockIdx.y;
    if (sp >= ninterps || j >= P) return;
    int s = j * C;
    int e = (j < P - 1) ? (s + C) : length;
    int nj = e - s;

    double *D = &Dg[(size_t)(sp * P + j) * (2 * m + 1) * Cmax];   // factor diagonal block (coupling rows zeroed)
    for (int br = 0; br <= 2 * m; ++br)
        for (int c = 0; c < nj; ++c)
        {
            int row = (s + c) + (br - m);
            D[br * nj + c] = (row >= s && row < e) ? QBAND(W, br, s + c, sp, ninterps, length) : 0.0;
        }
    banfac_local(D, m, nj);

    double *gj = &gg[(size_t)sp * length + s];                    // g_j = D_j^{-1} f_j
    for (int c = 0; c < nj; ++c) gj[c] = QRHS(B, s + c, sp, ninterps);
    banslv_local(D, m, nj, gj);

    if (j < P - 1)                                                // right spike: bottom m rows = Sup_j
        for (int b = 0; b < m; ++b)
        {
            double *col = &Vg[((size_t)(sp * P + j) * m + b) * Cmax];
            for (int c = 0; c < nj; ++c) col[c] = 0.0;
            for (int a = b; a < m; ++a) col[nj - m + a] = QBAND(W, a - b, e + b, sp, ninterps, length);
            banslv_local(D, m, nj, col);
        }
    if (j > 0)                                                    // left spike: top m rows = Sub_j
        for (int b = 0; b < m; ++b)
        {
            double *col = &Wg[((size_t)(sp * P + j) * m + b) * Cmax];
            for (int c = 0; c < nj; ++c) col[c] = 0.0;
            for (int a = 0; a <= b; ++a) col[a] = QBAND(W, 2 * m + a - b, s - m + b, sp, ninterps, length);
            banslv_local(D, m, nj, col);
        }
}

CUDA_KERNEL
void spike_reduced_kernel(double *gg, double *Vg, double *Wg, double *wrg, double *rrg,
                          int ninterps, int length, int C, int Cmax, int P, int m, int Mred, int R)
{
    int sp = blockIdx.x * blockDim.x + threadIdx.x;
    if (sp >= ninterps) return;
    double *wr = &wrg[(size_t)sp * (2 * Mred + 1) * R];
    double *rr = &rrg[(size_t)sp * R];
    for (size_t t = 0; t < (size_t)(2 * Mred + 1) * R; ++t) wr[t] = 0.0;

    for (int j = 0; j < P; ++j)
    {
        int s = j * C;
        int e = (j < P - 1) ? (s + C) : length;
        int nj = e - s;
        double *gj = &gg[(size_t)sp * length + s];
        double *Vj = &Vg[(size_t)(sp * P + j) * m * Cmax];      // Vj[b*Cmax + c]
        double *Wj = &Wg[(size_t)(sp * P + j) * m * Cmax];

        if (j >= 1)                                              // top equation
        {
            int p = qsp_pos_t(j, m);
            for (int a = 0; a < m; ++a) { wr[(size_t)Mred * R + (p + a)] += 1.0; rr[p + a] = gj[a]; }
            if (j < P - 1)
            {
                int cp = qsp_pos_t(j + 1, m);
                for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b)
                    wr[(size_t)(Mred + (p + a) - (cp + b)) * R + (cp + b)] += Vj[b * Cmax + a];
            }
            int cw = qsp_pos_b(j - 1, m);
            for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b)
                wr[(size_t)(Mred + (p + a) - (cw + b)) * R + (cw + b)] += Wj[b * Cmax + a];
        }
        if (j < P - 1)                                          // bottom equation
        {
            int p = qsp_pos_b(j, m);
            for (int a = 0; a < m; ++a) { wr[(size_t)Mred * R + (p + a)] += 1.0; rr[p + a] = gj[nj - m + a]; }
            int cv = qsp_pos_t(j + 1, m);
            for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b)
                wr[(size_t)(Mred + (p + a) - (cv + b)) * R + (cv + b)] += Vj[b * Cmax + (nj - m + a)];
            if (j >= 1)
            {
                int cw = qsp_pos_b(j - 1, m);
                for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b)
                    wr[(size_t)(Mred + (p + a) - (cw + b)) * R + (cw + b)] += Wj[b * Cmax + (nj - m + a)];
            }
        }
    }
    banfac_local(wr, Mred, R);
    banslv_local(wr, Mred, R, rr);                              // single-level (no GPU recursion yet)
}

CUDA_KERNEL
void spike_backsub_kernel(double *B, double *gg, double *Vg, double *Wg, double *rrg,
                          int ninterps, int length, int C, int Cmax, int P, int m, int R)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int sp = blockIdx.y;
    if (sp >= ninterps || j >= P) return;
    int s = j * C;
    int e = (j < P - 1) ? (s + C) : length;
    int nj = e - s;
    double *gj = &gg[(size_t)sp * length + s];
    double *Vj = &Vg[(size_t)(sp * P + j) * m * Cmax];
    double *Wj = &Wg[(size_t)(sp * P + j) * m * Cmax];
    double *rr = (R > 0) ? &rrg[(size_t)sp * R] : (double *)0;
    for (int c = 0; c < nj; ++c)
    {
        double x = gj[c];
        if (j < P - 1) { int cp = qsp_pos_t(j + 1, m); for (int b = 0; b < m; ++b) x -= Vj[b * Cmax + c] * rr[cp + b]; }
        if (j > 0)     { int cp = qsp_pos_b(j - 1, m); for (int b = 0; b < m; ++b) x -= Wj[b * Cmax + c] * rr[cp + b]; }
        QRHS(B, s + c, sp, ninterps) = x;
    }
}

// Host launcher (GPU). Allocates global scratch, runs the 3 phases, frees.
void quintic_spike_solve_gpu(double *W, double *B, int ninterps, int length, int chunk, int uniform)
{
    (void)uniform;                                  // Toeplitz cache is a GPU follow-up
    int m = QUINTIC_HALF_BAND;
    int C = (chunk > 0) ? chunk : 1024;
    if (C < 2 * m) C = 2 * m;
    if (C > length) C = length;
    int P = length / C; if (P < 1) P = 1;
    int Cmax = 2 * C;                               // last chunk absorbs remainder -> nj in [C, 2C)
    int Mred = 3 * m;
    int R = (P > 1) ? (2 * (P - 1) * m) : 0;
    int T = NUM_THREADS_INTERPOLATE;

    double *Dg, *gg, *Vg, *Wg, *wrg = 0, *rrg = 0;
    gpuErrchk(cudaMalloc(&Dg, (size_t)ninterps * P * (2 * m + 1) * Cmax * sizeof(double)));
    gpuErrchk(cudaMalloc(&gg, (size_t)ninterps * length * sizeof(double)));
    gpuErrchk(cudaMalloc(&Vg, (size_t)ninterps * P * m * Cmax * sizeof(double)));
    gpuErrchk(cudaMalloc(&Wg, (size_t)ninterps * P * m * Cmax * sizeof(double)));
    if (P > 1)
    {
        gpuErrchk(cudaMalloc(&wrg, (size_t)ninterps * (2 * Mred + 1) * R * sizeof(double)));
        gpuErrchk(cudaMalloc(&rrg, (size_t)ninterps * R * sizeof(double)));
    }

    dim3 grid((P + T - 1) / T, ninterps);
    spike_factor_kernel<<<grid, T>>>(W, B, Dg, gg, Vg, Wg, ninterps, length, C, Cmax, P, m);
    cudaDeviceSynchronize(); gpuErrchk(cudaGetLastError());
    if (P > 1)
    {
        spike_reduced_kernel<<<(ninterps + T - 1) / T, T>>>(gg, Vg, Wg, wrg, rrg, ninterps, length, C, Cmax, P, m, Mred, R);
        cudaDeviceSynchronize(); gpuErrchk(cudaGetLastError());
    }
    spike_backsub_kernel<<<grid, T>>>(B, gg, Vg, Wg, rrg, ninterps, length, C, Cmax, P, m, R);
    cudaDeviceSynchronize(); gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(Dg)); gpuErrchk(cudaFree(gg));
    gpuErrchk(cudaFree(Vg)); gpuErrchk(cudaFree(Wg));
    if (wrg) gpuErrchk(cudaFree(wrg));
    if (rrg) gpuErrchk(cudaFree(rrg));
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

#ifdef __CUDACC__
    (void)chunk; (void)uniform;   // GPU still uses the legacy solve (replaced in Task 5)
    int sblocks = std::ceil((ninterps + NUM_THREADS_INTERPOLATE - 1) / NUM_THREADS_INTERPOLATE);

    double *W;
    double *B;
    gpuErrchk(cudaMalloc(&W, band_count * sizeof(double)));
    gpuErrchk(cudaMalloc(&B, rhs_count * sizeof(double)));
    gpuErrchk(cudaMemset(W, 0, band_count * sizeof(double)));

    fill_quintic_band<<<ninterps, NUM_THREADS_INTERPOLATE>>>(x, y, W, B, ninterps, length);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    solve_quintic_band_batch<<<sblocks, NUM_THREADS_INTERPOLATE>>>(W, B, ninterps, length);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    set_quintic_constants<<<ninterps, NUM_THREADS_INTERPOLATE>>>(x, B, c1, c2, c3, c4, c5, ninterps, length);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(W));
    gpuErrchk(cudaFree(B));
#else
    double *W = new double[band_count]();   // value-initialized to 0
    double *B = new double[rhs_count];

    fill_quintic_band(x, y, W, B, ninterps, length);
    quintic_spike_solve(W, B, ninterps, length, chunk, uniform);   // SPIKE chunked solve
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



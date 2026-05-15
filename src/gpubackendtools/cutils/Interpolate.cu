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


CUDA_DEVICE
int CubicSpline::get_window(double x_new, int spline_index)
{
    int window = 0;
    // TODO: switch statement
    // TODO: MAKE FASTER
    if (spline_type == CUBIC_SPLINE_LINEAR_SPACING)
    {
        window = int(x_new / (x0[spline_index * length + 1] - x0[spline_index * length + 0]));
    }
    else if (spline_type == CUBIC_SPLINE_LOG10_SPACING)
    {
        window = int(log10(x_new) / (log10(x0[spline_index * length + 1]) - log10(x0[spline_index * length + 0])));  // does this slow it down?
    }
    else if (spline_type == CUBIC_SPLINE_GENERAL_SPACING)
    {
        window = binary_search(&x0[spline_index * length], 0, length, x_new);
        // printf("INSIDE: %e %e %e %d %d %d\n", x_new, x0[spline_index * length], x0[spline_index * length + length - 1], window, length, spline_index);
        // if (x0[spline_index * length + length - 1] == 0.0)
        // {
        //     printf("INSIDE2: %e %e %e %d %d %d\n", x_new, x0[spline_index * length], x0[spline_index * length + length - 1], window, length, spline_index);
        
        //     for (int j = 0; j < ninterps; j += 1)
        //     {
        //       for (int i = 0; i < length; i += 100)
        //       {
        //         printf("WHAT?: %d (%d) %d (%d) %.12e \n", j, ninterps, i, length, x0[j * length + i]);
        //       }
        //     }
        //     return -2;
        // }
      }
    else
    {
#ifdef __CUDACC__
        // printf("BAD cubic spline type. (%d)\n", spline_type);
#else
        std::string error_str = "BAD cubic spline type. (" + std::to_string(spline_type) + ")"; 
        throw std::invalid_argument(error_str);
#endif // __CUDACC__
    }

    if ((window < 0) || (window >= length))
    {
#ifdef __CUDACC__
        // printf("Outside spline. Using edge value.");
        if (window < 0) window = 0;
        if (window >= length) window = length - 1;
#else
        std::string error_str = "Outside spline." + std::to_string(window) + " " + std::to_string(length) + " " + std::to_string(x_new) + " " + std::to_string(x0[length-1]); 
        throw std::invalid_argument(error_str);
#endif // __CUDACC__
    }
    
    return window;
}

CUDA_DEVICE
CubicSplineSegment CubicSpline::get_cublic_spline_segment(double x_new, int spline_index)
{
    int window = get_window(x_new, spline_index); 
    if (window == -2)
    {
      // printf("OUTSIDE: %e %e %e %d %d %d\n", x_new, x0[spline_index * length], x0[spline_index * length + length - 1], window, length, spline_index);
#ifdef __CUDACC__
#else
      throw std::invalid_argument("BAD.");
#endif
    }    
    int _index = spline_index * length + window; 
    CubicSplineSegment segment(x0[_index], y0[_index], c1[_index], c2[_index], c3[_index], spline_type);
    return segment;
}


CUDA_DEVICE
double CubicSpline::eval_single(double x_new, int spline_index)
{
    CubicSplineSegment segment = get_cublic_spline_segment(x_new, spline_index);
    return segment.eval(x_new);
}


CUDA_DEVICE
void CubicSpline::eval(double *y_new, double *x_new, int *spline_index, int N)
{
#ifdef __CUDACC__
  int start1 = threadIdx.x + blockIdx.x * blockDim.x;
  int diff1 = gridDim.x * blockDim.x;
#else
  int start1 = 0;
  int diff1 = 1;
#endif
    for (int i = start1; i < N; i += diff1)
    {
        y_new[i] = eval_single(x_new[i], spline_index[i]);
    }
}

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

CUDA_DEVICE
int CubicSpline::even_sampled_search(double *array, int nmin, int nmax, double x) {
    // TODO: adjust this. At specialized gpu array searching. 
    double dx = array[1] - array[0];
    return (int)floor(x/dx);
}

// Recursive binary search function.
// Return nearest smaller neighbor of x in array[nmin,nmax] is present,
// otherwise -1
CUDA_DEVICE
int CubicSpline::binary_search(double *array, int nmin, int nmax, double x)
{
    // catch if x exactly matches array[nmin]
    if(x==array[nmin]) return nmin;
    
    int next;
    if(nmax>nmin)
    {
        int mid = nmin + (nmax - nmin) / 2;
        
        //find next unique element of array
        next = mid;
        while(array[mid]==array[next]) next++;
        
        // If the element is present at the middle
        // itself
        if (x > array[mid] && x < array[next])
            return mid;
        
        // the element is in the lower half
        if (array[mid] >= x)
            return binary_search(array, nmin, mid, x);
        
        // the element is in upper half
        return binary_search(array, next, nmax, x);
    }
    
    // We reach here when element is not
    // present in array
    return -1;
}



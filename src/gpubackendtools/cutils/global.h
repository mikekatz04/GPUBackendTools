/*  
 */

#ifndef _GLOBAL_HEADER_
#define _GLOBAL_HEADER_

#include <stdio.h>
#include <stdlib.h>

#include <complex>

#include "cuda_complex.hpp"

#ifdef __CUDACC__
#include "cuda_runtime_api.h"

#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_DEVICE __device__
#define CUDA_KERNEL __global__
#define CUDA_SHARED __shared__
#define CUDA_SYNCTHREADS __syncthreads();
#define CUDA_THREAD_ZERO (threadIdx.x == 0)

/*
Function for gpu Error checking.
//*/
#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_DEVICE
#define CUDA_KERNEL
#define CUDA_SHARED
#define CUDA_SYNCTHREADS
#define CUDA_THREAD_ZERO (true)
#endif

typedef gcmplx::complex<double> cmplx;

#define invsqrt2 0.7071067811865475
#define invsqrt3 0.5773502691896258
#define invsqrt6 0.4082482904638631

#endif

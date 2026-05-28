#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#include "cuda_complex.hpp"
#include "stdio.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_DEVICE __device__
#define CUDA_KERNEL __global__
#define CUDA_SHARED __shared__
#define CUDA_SYNC_THREADS __syncthreads();
#define THREAD_ZERO (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
// Per-axis thread / block indexing. Kernels MUST pick which axis each
// loop maps to and use the matching macro -- there is no un-suffixed
// fallback (the previous BLOCK_START / GRID_INCR / THREAD_START /
// BLOCK_INCR names were removed when the 3D-grid refactor landed; see
// the WDM chunked-het kernels in lisa-on-gpu for the canonical
// (chunks on Z, binaries on X, pixels on threads.X) usage).
#define THREAD_START_X threadIdx.x
#define BLOCK_INCR_X   blockDim.x
#define BLOCK_START_X  blockIdx.x
#define GRID_INCR_X    gridDim.x
#define THREAD_START_Y threadIdx.y
#define BLOCK_INCR_Y   blockDim.y
#define BLOCK_START_Y  blockIdx.y
#define GRID_INCR_Y    gridDim.y
#define THREAD_START_Z threadIdx.z
#define BLOCK_INCR_Z   blockDim.z
#define BLOCK_START_Z  blockIdx.z
#define GRID_INCR_Z    gridDim.z

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_DEVICE
#define CUDA_KERNEL
#define CUDA_SHARED
#define CUDA_SYNC_THREADS
#define THREAD_ZERO (true)
// CPU stubs collapse every axis to the trivial single-element loop
// (start=0, increment=1). Any kernel that "loops" over an axis on GPU
// degenerates to a plain for-loop on CPU.
#define THREAD_START_X 0
#define BLOCK_INCR_X   1
#define BLOCK_START_X  0
#define GRID_INCR_X    1
#define THREAD_START_Y 0
#define BLOCK_INCR_Y   1
#define BLOCK_START_Y  0
#define GRID_INCR_Y    1
#define THREAD_START_Z 0
#define BLOCK_INCR_Z   1
#define BLOCK_START_Z  0
#define GRID_INCR_Z    1

#endif


typedef gcmplx::complex<double> cmplx;

#endif // __GLOBAL_H__

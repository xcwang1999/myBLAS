#ifndef MYBLAS_INCLUDE_HGEMM_KERNELS_CUH_
#define MYBLAS_INCLUDE_HGEMM_KERNELS_CUH_
#include <cuda_fp16.h>
__global__ void hgemmKernel1(const int M, const int N, const int K,
                             half *__restrict__ A, const size_t pitchA,
                             half *__restrict__ B, const size_t pitchB,
                             half *__restrict__ C, const size_t pitchC);

__global__ void hgemmKernel2(const int M, const int N, const int K,
                             half *__restrict__ A, const size_t pitchA,
                             half *__restrict__ B, const size_t pitchB,
                             half *__restrict__ C, const size_t pitchC);

#endif  // MYBLAS_INCLUDE_HGEMM_KERNELS_CUH_

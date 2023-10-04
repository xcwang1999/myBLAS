#ifndef MYBLAS_INCLUDE_SGEMM_KERNELS_CUH_
#define MYBLAS_INCLUDE_SGEMM_KERNELS_CUH_

__global__ void sgemmKernel1(const int M, const int N, const int K,
                             const float *__restrict__ A,
                             const float *__restrict__ B,
                             float *__restrict__ C);
__global__ void sgemmKernel2(const int M, const int N, const int K,
                             float *__restrict__ A, float *__restrict__ B,
                             float *__restrict__ C);
__global__ void sgemmKernel3(const int M, const int N, const int K,
                             float *__restrict__ A, float *__restrict__ B,
                             float *__restrict__ C);
__global__ void sgemmKernel4(const int M, const int N, const int K,
                             float *__restrict__ A, float *__restrict__ B,
                             float *__restrict__ C);
__global__ void sgemmKernel5(const int M, const int N, const int K,
                             float *__restrict__ A, float *__restrict__ B,
                             float *__restrict__ C);
__global__ void sgemmKernel6(const int M, const int N, const int K,
                             const float *__restrict__ A,
                             const float *__restrict__ B,
                             float *__restrict__ C);
__global__ void sgemmKernel7(const int M, const int K, float *__restrict__ A,
                             const size_t pitchA, float *__restrict__ B,
                             const size_t pitchB, float *__restrict__ C,
                             const size_t pitchC);

#endif  // MYBLAS_INCLUDE_SGEMM_KERNELS_CUH_
#include<cuda_fp16.h>
template <typename T>
__global__ void nativeMatmul1(T *result, const T *lhs, const T *rhs,
                              int lhsNRows, int rhsNCols, int lhsNCols) {
  int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
  int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

  if (globalRow < lhsNRows && globalCol < rhsNCols) {
    T tempSum = 0.0;
    for (int k = 0; k < lhsNCols; k++) {
      tempSum += lhs[globalRow * lhsNCols + k] * rhs[k * rhsNCols + globalCol];
    }
    result[globalRow * rhsNCols + globalCol] = tempSum;
  }
}

template <typename T>
__global__ void nativeMatmul2(T *C, const T *A, const T *B, int M, int N,
                              int K) {
  size_t blockSize = blockDim.x * blockDim.y;
  __shared__ T sharedMemory[2 * 32 * 32];
  T *sharedM = sharedMemory;
  T *sharedN = sharedMemory + blockSize;

  int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
  int globalCol = blockIdx.x * blockDim.x + threadIdx.x;
  int globalBlockTRow = blockIdx.x * blockDim.x + threadIdx.y;
  //    int globalBlockTCol = blockIdx.y * blockDim.y + threadIdx.x;
  int indexInBlock = threadIdx.y * blockDim.x + threadIdx.x;

  T tempSum;

  if (globalRow < M && globalCol < K)
    sharedM[indexInBlock] = A[globalRow * K + globalCol];
  else
    sharedM[indexInBlock] = 0.0;

  for (int i = 0; i < ((N - 1) / blockDim.x + 1); i++) {
    tempSum = 0.0;
    if (globalBlockTRow < K && (i * blockDim.y + threadIdx.x) <
                                   N)  // Suppose 'K' is equal to 'rhsNRows'.
      sharedN[indexInBlock] =
          B[globalBlockTRow * N + i * blockDim.y + threadIdx.x];
    else
      sharedN[indexInBlock] = 0.0;

    __syncthreads();
    for (int k = 0; k < blockDim.x; k++)
      tempSum += sharedM[threadIdx.y * blockDim.x + k] *
                 sharedN[k * blockDim.x + threadIdx.x];

    __syncthreads();

    if (globalRow < M && (threadIdx.x + i * blockDim.x) < N)
      atomicAdd(&C[globalRow * N + (threadIdx.x + i * blockDim.x)], tempSum);
  }
}

template <typename T>
__global__ void nativeMatmul3(T *C, const T *A, const T *B, int M, int N,
                              int K) {
  size_t blockSize = blockDim.x * blockDim.y;
  __shared__ T sharedMemory[2 * 32 * 32];
  T *sharedM = sharedMemory;
  T *sharedN = sharedMemory + blockSize;

  int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
  int globalCol = blockIdx.x * blockDim.x + threadIdx.x;
  int indexInBlock = threadIdx.y * blockDim.x + threadIdx.x;

  T tempSum = 0.0;

  for (int i = 0; i < ((K - 1) / blockDim.x + 1); i++) {
    if (i * blockDim.x + threadIdx.x < K && globalRow < M)
      sharedM[indexInBlock] = A[globalRow * K + i * blockDim.x + threadIdx.x];
    else
      sharedM[indexInBlock] = 0.0;
    if (i * blockDim.y + threadIdx.y < K &&
        globalCol < N)  // Suppose 'K' is equal to 'rhsNRows'.
      sharedN[indexInBlock] = B[(i * blockDim.y + threadIdx.y) * N + globalCol];
    else
      sharedN[indexInBlock] = 0.0;
    __syncthreads();
    for (int j = 0; j < blockDim.x; j++)
      tempSum += sharedM[threadIdx.y * blockDim.x + j] *
                 sharedN[j * blockDim.x + threadIdx.x];
    __syncthreads();
  }
  if (globalRow < M && globalCol < N) C[globalRow * N + globalCol] = tempSum;
}

template __global__ void nativeMatmul1<half>(half *result,
                                                     const half *lhs,
                                                     const half *rhs,
                                                     int lhsNRows, int rhsNCols,
                                                     int lhsNCols);
template __global__ void nativeMatmul2<half>(half *result,
                                                     const half *lhs,
                                                     const half *rhs,
                                                     int lhsNRows, int rhsNCols,
                                                     int lhsNCols);
template __global__ void nativeMatmul3<half>(half *result,
                                                     const half *lhs,
                                                     const half *rhs,
                                                     int lhsNRows, int rhsNCols,
                                                     int lhsNCols);
template __global__ void nativeMatmul1<float>(float *result, const float *lhs,
                                              const float *rhs, int lhsNRows,
                                              int rhsNCols, int lhsNCols);
template __global__ void nativeMatmul2<float>(float *result, const float *lhs,
                                              const float *rhs, int lhsNRows,
                                              int rhsNCols, int lhsNCols);
template __global__ void nativeMatmul3<float>(float *result, const float *lhs,
                                              const float *rhs, int lhsNRows,
                                              int rhsNCols, int lhsNCols);
template __global__ void nativeMatmul1<double>(double *result,
                                               const double *lhs,
                                               const double *rhs, int lhsNRows,
                                               int rhsNCols, int lhsNCols);
template __global__ void nativeMatmul2<double>(double *result,
                                               const double *lhs,
                                               const double *rhs, int lhsNRows,
                                               int rhsNCols, int lhsNCols);
template __global__ void nativeMatmul3<double>(double *result,
                                               const double *lhs,
                                               const double *rhs, int lhsNRows,
                                               int rhsNCols, int lhsNCols);
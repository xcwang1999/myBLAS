#include <cuda_fp16.h>

namespace myblas::transpose {

template <typename T>
__global__ void transposeKernel(T *inputMatrix, const int nRows,
                                const int nCols) {
  __shared__ T sharedM[32 * 32];
  int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
  int globalCol = blockIdx.x * blockDim.x + threadIdx.x;
  int globalBlockTRow = blockIdx.x * blockDim.x + threadIdx.y;
  int globalBlockTCol = blockIdx.y * blockDim.y + threadIdx.x;
  int indexInBlock = threadIdx.y * blockDim.x + threadIdx.x;
  int indexInBlockT = threadIdx.x * blockDim.x + threadIdx.y;

  if (globalRow < nRows && globalCol < nCols)
    sharedM[indexInBlock] = inputMatrix[globalRow * nCols + globalCol];
  __syncthreads();

  if (globalBlockTRow < nCols && globalBlockTCol < nRows)
    inputMatrix[globalBlockTRow * nRows + globalBlockTCol] =
        sharedM[indexInBlockT];
}

template __global__ void transposeKernel<half>(half *inputMatrix,
                                               const int nRows,
                                               const int nCols);
template __global__ void transposeKernel<float>(float *inputMatrix,
                                                const int nRows,
                                                const int nCols);
template __global__ void transposeKernel<double>(double *inputMatrix,
                                                 const int nRows,
                                                 const int nCols);

template <typename T>
void transpose(T *inputMatrix, const int m, const int n) {
  constexpr dim3 block_size(32, 32);
  const dim3 grid_size((n - 1) / block_size.x + 1, (m - 1) / block_size.y + 1);
  transposeKernel<T><<<grid_size, block_size>>>(inputMatrix, m, n);
}

template void transpose<half>(half *inputMatrix, const int nRows,
                              const int nCols);
template void transpose<float>(float *inputMatrix, const int nRows,
                               const int nCols);
template void transpose<double>(double *inputMatrix, const int nRows,
                                const int nCols);

}  // namespace myblas::transpose
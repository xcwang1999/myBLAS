#include <cuda_fp16.h>
template <typename T>
__global__ void transposeKernel(T *inputMatrix, int nRows, int nCols) {
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

template __global__ void transposeKernel<half>(half *inputMatrix, int nRows,
                                                int nCols);
template __global__ void transposeKernel<float>(float *inputMatrix, int nRows,
                                                int nCols);
template __global__ void transposeKernel<double>(double *inputMatrix, int nRows,
                                                 int nCols);
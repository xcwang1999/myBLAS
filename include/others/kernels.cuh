#ifndef MYBLAS_INCLUDE_OTHERS_KERNELS_CUH_
#define MYBLAS_INCLUDE_OTHERS_KERNELS_CUH_
#include <cuda_fp16.h>
template <typename T>
__global__ void transposeKernel(T *inputMatrix, int nRows, int nCols);

extern template __global__ void transposeKernel<half>(half *inputMatrix, int nRows,
                                                int nCols);
extern template __global__ void transposeKernel<float>(float *inputMatrix,
                                                       int nRows, int nCols);
extern template __global__ void transposeKernel<double>(double *inputMatrix,
                                                        int nRows, int nCols);
template <typename T>
__global__ void nativeMatmul1(T *result, const T *lhs, const T *rhs,
                              int lhsNRows, int rhsNCols, int lhsNCols);
template <typename T>
__global__ void nativeMatmul2(T *result, const T *lhs, const T *rhs,
                              int lhsNRows, int rhsNCols, int lhsNCols);
template <typename T>
__global__ void nativeMatmul3(T *result, const T *lhs, const T *rhs,
                              int lhsNRows, int rhsNCols, int lhsNCols);

extern template __global__ void nativeMatmul1<half>(half *result,
                                                     const half *lhs,
                                                     const half *rhs,
                                                     int lhsNRows, int rhsNCols,
                                                     int lhsNCols);
extern template __global__ void nativeMatmul2<half>(half *result,
                                                     const half *lhs,
                                                     const half *rhs,
                                                     int lhsNRows, int rhsNCols,
                                                     int lhsNCols);
extern template __global__ void nativeMatmul3<half>(half *result,
                                                     const half *lhs,
                                                     const half *rhs,
                                                     int lhsNRows, int rhsNCols,
                                                     int lhsNCols);

extern template __global__ void nativeMatmul1<float>(float *result,
                                                     const float *lhs,
                                                     const float *rhs,
                                                     int lhsNRows, int rhsNCols,
                                                     int lhsNCols);
extern template __global__ void nativeMatmul2<float>(float *result,
                                                     const float *lhs,
                                                     const float *rhs,
                                                     int lhsNRows, int rhsNCols,
                                                     int lhsNCols);
extern template __global__ void nativeMatmul3<float>(float *result,
                                                     const float *lhs,
                                                     const float *rhs,
                                                     int lhsNRows, int rhsNCols,
                                                     int lhsNCols);
extern template __global__ void nativeMatmul1<double>(
    double *result, const double *lhs, const double *rhs, int lhsNRows,
    int rhsNCols, int lhsNCols);
extern template __global__ void nativeMatmul2<double>(
    double *result, const double *lhs, const double *rhs, int lhsNRows,
    int rhsNCols, int lhsNCols);
extern template __global__ void nativeMatmul3<double>(
    double *result, const double *lhs, const double *rhs, int lhsNRows,
    int rhsNCols, int lhsNCols);

#endif  // MYBLAS_INCLUDE_OTHERS_KERNELS_CUH_
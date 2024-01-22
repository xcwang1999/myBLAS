#pragma once
#include <cuda_fp16.h>

namespace myblas::transpose {
template <typename T>
void transpose(T *inputMatrix, const int m, const int n);

extern template void transpose<half>(half *inputMatrix, const int nRows,
                                     const int nCols);
extern template void transpose<float>(float *inputMatrix, const int nRows,
                                      const int nCols);
extern template void transpose<double>(double *inputMatrix, const int nRows,
                                       const int nCols);

} // namespace myblas::transpose
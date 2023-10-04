#include <iostream>

#include "hgemm/kernels.cuh"
#include "matrix/matrixGPU.cuh"
#include "others/kernels.cuh"
#include "sgemm/kernels.cuh"
#include "utils.h"

template class MatrixGPU<float>;
template class MatrixGPU<half>;

template <class T>
MatrixGPU<T>::MatrixGPU() noexcept {
  m_nRows = 0;
  m_nCols = 0;
  m_nElements = 0;
  m_matrixValue = nullptr;
}

template <class T>
MatrixGPU<T>::MatrixGPU(int nRows, int nCols) noexcept {
  m_nRows = nRows;
  m_nCols = nCols;
  m_nElements = m_nRows * m_nCols;
  CHECK_ERROR(cudaMalloc((void **)&m_matrixValue, m_nElements * sizeof(T)));
  CHECK_ERROR(cudaMemset(m_matrixValue, 0, m_nElements * sizeof(T)));
}

template <class T>
MatrixGPU<T>::MatrixGPU(int nRows, int nCols, T *matrixValue) noexcept {
  m_nRows = nRows;
  m_nCols = nCols;
  m_nElements = m_nRows * m_nCols;
  CHECK_ERROR(cudaMalloc((void **)&m_matrixValue, m_nElements * sizeof(T)));
  CHECK_ERROR(cudaMemcpy(m_matrixValue, matrixValue, m_nElements * sizeof(T),
                         cudaMemcpyHostToDevice));
}

template <class T>
MatrixGPU<T>::MatrixGPU(const MatrixGPU<T> &inputMatrix) noexcept {
  if (inputMatrix.m_matrixValue != nullptr) {
    m_nRows = inputMatrix.m_nRows;
    m_nCols = inputMatrix.m_nCols;
    m_nElements = m_nRows * m_nCols;
    CHECK_ERROR(cudaMalloc((void **)&m_matrixValue, m_nElements * sizeof(T)));
    CHECK_ERROR(cudaMemcpy(m_matrixValue, inputMatrix.m_matrixValue,
                           m_nElements * sizeof(T), cudaMemcpyDeviceToDevice));
  } else {
    m_nRows = 0;
    m_nCols = 0;
    m_nElements = 0;
    m_matrixValue = nullptr;
  }
}

template <class T>
MatrixGPU<T>::MatrixGPU(MatrixGPU<T> &&inputMatrix) noexcept {
  if (inputMatrix.m_matrixValue != nullptr) {
    m_nRows = inputMatrix.m_nRows;
    m_nCols = inputMatrix.m_nCols;
    m_nElements = m_nRows * m_nCols;
    m_matrixValue = inputMatrix.m_matrixValue;
    inputMatrix.m_matrixValue = nullptr;
  } else {
    m_nRows = 0;
    m_nCols = 0;
    m_nElements = 0;
    m_matrixValue = nullptr;
  }
}

template <class T>
MatrixGPU<T>::~MatrixGPU() noexcept {
  if (m_matrixValue != nullptr) {
    CHECK_ERROR(cudaFree(m_matrixValue));
    m_matrixValue = nullptr;
  }
}

template <class T>
MatrixGPU<T> &MatrixGPU<T>::operator=(const MatrixGPU<T> &rhs) noexcept {
  if (this == &rhs) return *this;
  if (rhs.m_matrixValue == nullptr) {
    if (m_matrixValue != nullptr) {
      CHECK_ERROR(cudaFree(m_matrixValue));
      m_matrixValue = nullptr;
    }
  } else {
    m_nRows = rhs.m_nRows;
    m_nCols = rhs.m_nCols;
    m_nElements = m_nRows * m_nCols;
    if (m_matrixValue != nullptr) cudaFree(m_matrixValue);
    CHECK_ERROR(cudaMalloc((void **)&m_matrixValue, m_nElements * sizeof(T)));
    CHECK_ERROR(cudaMemcpy(m_matrixValue, rhs.m_matrixValue,
                           m_nElements * sizeof(T), cudaMemcpyDeviceToDevice));
  }

  return *this;
}

template <class T>
MatrixGPU<T> &MatrixGPU<T>::operator=(MatrixGPU<T> &&rhs) noexcept {
  if (rhs.m_matrixValue == nullptr) {
    if (m_matrixValue != nullptr) {
      CHECK_ERROR(cudaFree(m_matrixValue));
      m_matrixValue = nullptr;
    }
  } else {
    m_nRows = rhs.m_nRows;
    m_nCols = rhs.m_nCols;
    m_nElements = m_nRows * m_nCols;
    if (m_matrixValue != nullptr) cudaFree(m_matrixValue);
    m_matrixValue = rhs.m_matrixValue;
    rhs.m_matrixValue = nullptr;
  }

  return *this;
}

template <class T>
void MatrixGPU<T>::printMatrix() {
  ASSERT(m_matrixValue != nullptr, "printValue warning: No value to output.\n");
  T *tempMatrix = new T[m_nElements];
  CHECK_ERROR(cudaMemcpy(tempMatrix, m_matrixValue, m_nElements * sizeof(T),
                         cudaMemcpyDeviceToHost));
  printValue<T>(m_nRows, m_nCols, tempMatrix);
  delete[] tempMatrix;
}

template <>
inline void MatrixGPU<half>::printMatrix() {
  ASSERT(m_matrixValue != nullptr, "printValue warning: No value to output.\n");
  half *tmp_half = new half[m_nElements];
  CHECK_ERROR(cudaMemcpy(tmp_half, m_matrixValue, m_nElements * sizeof(half),
                         cudaMemcpyDeviceToHost));
  float *tmp_float = new float[m_nElements];
  for (int n = 0; n < m_nElements; n++) {
    tmp_float[n] = __half2float(tmp_half[n]);
  }
  printValue<float>(m_nRows, m_nCols, tmp_float);
  delete[] tmp_half;
  delete[] tmp_float;
}

template <class T>
T *MatrixGPU<T>::getMatrix() {
  T *matrixValueHost = new T[m_nElements * sizeof(T)];
  cudaMemcpy(matrixValueHost, m_matrixValue, m_nElements * sizeof(T),
             cudaMemcpyDeviceToHost);
  return matrixValueHost;
}

template <class T>
void MatrixGPU<T>::ones(int nRows, int nCols) {
  m_nRows = nRows;
  m_nCols = nCols;
  m_nElements = nRows * nCols;
  T *tempMatrix = new T[m_nElements];
  for (int n = 0; n < m_nElements; n++) {
    tempMatrix[n] = 1;
  }
  CHECK_ERROR(cudaMalloc((void **)&m_matrixValue, m_nElements * sizeof(T)));
  CHECK_ERROR(cudaMemcpy(m_matrixValue, tempMatrix, m_nElements * sizeof(T),
                         cudaMemcpyHostToDevice));
  delete[] tempMatrix;
}

template <class T>
void MatrixGPU<T>::zeros(int nRows, int nCols) {
  m_nRows = nRows;
  m_nCols = nCols;
  m_nElements = nRows * nCols;
  CHECK_ERROR(cudaMalloc((void **)&m_matrixValue, m_nElements * sizeof(T)));
  CHECK_ERROR(cudaMemset(m_matrixValue, 0, m_nElements * sizeof(T)));
}

template <class T>
MatrixGPU<T> MatrixGPU<T>::matmul(const MatrixGPU<T> &rhs) {
  int lhsNRows = (*this).m_nRows;
  int lhsNCols = (*this).m_nCols;
  int rhsNRows = rhs.m_nRows;
  int rhsNCols = rhs.m_nCols;
  ASSERT((*this).m_matrixValue != nullptr && rhs.m_matrixValue != nullptr,
         "matmul error: invalid matrix.\n")
  ASSERT(lhsNCols == rhsNRows, "matmul error: Matrix size mismatch.\n")
  MatrixGPU<T> result(lhsNRows, rhsNCols);

  dim3 block_size(32, 32);
  dim3 grid_size((rhsNCols - 1) / block_size.x + 1,
                 (lhsNRows - 1) / block_size.y + 1);  // native
  nativeMatmul1<T><<<grid_size, block_size>>>(
      result.m_matrixValue, (*this).m_matrixValue, rhs.m_matrixValue, lhsNRows,
      rhsNCols, lhsNCols);

  // dim3 block_size(32, 32);
  // dim3 grid_size((lhsNCols - 1) / block_size.x + 1,
  //                (lhsNRows - 1) / block_size.y + 1);  // dot
  // nativeMatmul2<T>
  //     <<<grid_size, block_size>>>(
  //         result.m_matrixValue, (*this).m_matrixValue, rhs.m_matrixValue,
  //         lhsNRows, rhsNCols, lhsNCols);

  // dim3 block_size(32, 32);
  // dim3 grid_size((rhsNCols - 1) / block_size.x + 1,
  //                (lhsNRows - 1) / block_size.y + 1);  // outer
  // nativeMatmul3<T>
  //     <<<grid_size, block_size>>>(
  //         result.m_matrixValue, (*this).m_matrixValue, rhs.m_matrixValue,
  //         lhsNRows, rhsNCols, lhsNCols);

  CHECK_ERROR(cudaGetLastError());
  CHECK_ERROR(cudaDeviceSynchronize());
  return result;
}

template <>
MatrixGPU<float> MatrixGPU<float>::sgemm(const MatrixGPU<float> &rhs) {
  int lhsNRows = (*this).m_nRows;
  int lhsNCols = (*this).m_nCols;
  int rhsNRows = rhs.m_nRows;
  int rhsNCols = rhs.m_nCols;
  ASSERT((*this).m_matrixValue != nullptr && rhs.m_matrixValue != nullptr,
         "matmul error: invalid matrix.\n")
  ASSERT(lhsNCols == rhsNRows, "matmul error: Matrix size mismatch.\n")
  MatrixGPU<float> result(lhsNRows, rhsNCols);

  const int BM = 128;
  const int BN = 128;
  const int TM = 8;
  const int TN = 8;
  dim3 block_size(BM * BN / (TM * TN));
  dim3 grid_size((rhsNCols - 1) / BN + 1, (lhsNRows - 1) / BM + 1);
  sgemmKernel1<<<grid_size, block_size>>>(
      lhsNRows, rhsNCols, lhsNCols, (*this).m_matrixValue, rhs.m_matrixValue,
      result.m_matrixValue);
  CHECK_ERROR(cudaGetLastError());
  CHECK_ERROR(cudaDeviceSynchronize());
  return result;
}

template <class T>
MatrixGPU<T> &MatrixGPU<T>::transpose() {
  ASSERT(m_matrixValue != nullptr, "transpose error: No value to transpose.\n")
  dim3 block_size(32, 32);
  dim3 grid_size((m_nCols - 1) / block_size.x + 1,
                 (m_nRows - 1) / block_size.y + 1);
  transposeKernel<T>
      <<<grid_size, block_size>>>(m_matrixValue, m_nRows, m_nCols);
  CHECK_ERROR(cudaGetLastError());
  CHECK_ERROR(cudaDeviceSynchronize());
  T temp = m_nRows;
  m_nRows = m_nCols;
  m_nCols = temp;
  return *this;
}

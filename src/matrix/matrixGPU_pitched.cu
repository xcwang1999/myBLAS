#include <iostream>

#include "hgemm/kernels.cuh"
#include "matrix/matrixGPU.cuh"
#include "sgemm/kernels.cuh"
#include "utils.h"

template class MatrixGPUPitched<float>;
template class MatrixGPUPitched<half>;

template <class T>
MatrixGPUPitched<T>::MatrixGPUPitched() noexcept {
  m_nRows = 0;
  m_nCols = 0;
  m_nElements = 0;
  m_matrixValue = nullptr;
}

template <class T>
MatrixGPUPitched<T>::MatrixGPUPitched(int nRows, int nCols) noexcept {
  m_nRows = nRows;
  m_nCols = nCols;
  m_nElements = m_nRows * m_nCols;
  CHECK_ERROR(cudaMallocPitch((void **)&m_matrixValue, &m_pitch,
                              m_nCols * sizeof(T), m_nRows));
  CHECK_ERROR(
      cudaMemset2D(m_matrixValue, m_pitch, 0, m_nCols * sizeof(T), m_nRows));
}

template <class T>
MatrixGPUPitched<T>::MatrixGPUPitched(int nRows, int nCols,
                                      T *matrixValue) noexcept {
  m_nRows = nRows;
  m_nCols = nCols;
  m_nElements = m_nRows * m_nCols;
  CHECK_ERROR(cudaMallocPitch((void **)&m_matrixValue, &m_pitch,
                              m_nCols * sizeof(T), m_nRows));
  CHECK_ERROR(
      cudaMemset2D(m_matrixValue, m_pitch, 0, m_nCols * sizeof(T), m_nRows));
  CHECK_ERROR(cudaMemcpy2D(m_matrixValue, m_pitch, matrixValue,
                           nCols * sizeof(T), m_nCols * sizeof(T), m_nRows,
                           cudaMemcpyHostToDevice));
}

template <class T>
MatrixGPUPitched<T>::MatrixGPUPitched(
    const MatrixGPUPitched<T> &inputMatrix) noexcept {
  if (inputMatrix.m_matrixValue != nullptr) {
    m_nRows = inputMatrix.m_nRows;
    m_nCols = inputMatrix.m_nCols;
    m_nElements = m_nRows * m_nCols;
    CHECK_ERROR(cudaMallocPitch((void **)&m_matrixValue, &m_pitch,
                                m_nCols * sizeof(T), m_nRows));
    CHECK_ERROR(
        cudaMemset2D(m_matrixValue, m_pitch, 0, m_nCols * sizeof(T), m_nRows));
    CHECK_ERROR(cudaMemcpy2D(m_matrixValue, m_pitch, inputMatrix.m_matrixValue,
                             inputMatrix.m_pitch, m_nCols * sizeof(T), m_nRows,
                             cudaMemcpyDeviceToDevice));
  } else {
    m_nRows = 0;
    m_nCols = 0;
    m_nElements = 0;
    m_matrixValue = nullptr;
  }
}

template <class T>
MatrixGPUPitched<T>::MatrixGPUPitched(
    MatrixGPUPitched<T> &&inputMatrix) noexcept {
  if (inputMatrix.m_matrixValue != nullptr) {
    m_nRows = inputMatrix.m_nRows;
    m_nCols = inputMatrix.m_nCols;
    m_pitch = inputMatrix.m_pitch;
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
MatrixGPUPitched<T>::~MatrixGPUPitched() noexcept {
  if (m_matrixValue != nullptr) {
    CHECK_ERROR(cudaFree(m_matrixValue));
    m_matrixValue = nullptr;
  }
}

template <class T>
MatrixGPUPitched<T> &MatrixGPUPitched<T>::operator=(
    const MatrixGPUPitched<T> &rhs) noexcept {
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
    CHECK_ERROR(cudaMallocPitch((void **)&m_matrixValue, &m_pitch,
                                m_nCols * sizeof(T), m_nRows));
    CHECK_ERROR(
        cudaMemset2D(m_matrixValue, m_pitch, 0, m_nCols * sizeof(T), m_nRows));
    CHECK_ERROR(cudaMemcpy2D(m_matrixValue, m_pitch, rhs.m_matrixValue,
                             rhs.m_pitch, m_nCols * sizeof(T), m_nRows,
                             cudaMemcpyDeviceToDevice));
  }

  return *this;
}

template <class T>
MatrixGPUPitched<T> &MatrixGPUPitched<T>::operator=(
    MatrixGPUPitched<T> &&rhs) noexcept {
  if (rhs.m_matrixValue == nullptr) {
    if (m_matrixValue != nullptr) {
      CHECK_ERROR(cudaFree(m_matrixValue));
      m_matrixValue = nullptr;
    }
  } else {
    m_nRows = rhs.m_nRows;
    m_nCols = rhs.m_nCols;
    m_pitch = rhs.m_pitch;
    m_nElements = m_nRows * m_nCols;
    if (m_matrixValue != nullptr) CHECK_ERROR(cudaFree(m_matrixValue));
    m_matrixValue = rhs.m_matrixValue;
    rhs.m_matrixValue = nullptr;
  }

  return *this;
}

template <class T>
void MatrixGPUPitched<T>::printMatrix() {
  ASSERT(m_matrixValue != nullptr, "printValue warning: No value to output.\n");
  T *tempMatrix = new T[m_nElements];
  CHECK_ERROR(cudaMemcpy(tempMatrix, m_matrixValue, m_nElements * sizeof(T),
                         cudaMemcpyDeviceToHost));
  printValue<T>(m_nRows, m_nCols, tempMatrix);
  delete[] tempMatrix;
}

template <>
inline void MatrixGPUPitched<half>::printMatrix() {
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
T *MatrixGPUPitched<T>::getMatrix() {
  T *matrixValueHost = new T[m_nElements * sizeof(T)];
  CHECK_ERROR(cudaMemcpy2D(matrixValueHost, m_nCols * sizeof(T), m_matrixValue,
                           m_pitch, m_nCols * sizeof(T), m_nRows,
                           cudaMemcpyDeviceToHost));
  return matrixValueHost;
}

template <class T>
void MatrixGPUPitched<T>::zeros(int nRows, int nCols) {
  m_nRows = nRows;
  m_nCols = nCols;
  m_nElements = nRows * nCols;
  CHECK_ERROR(cudaMallocPitch((void **)&m_matrixValue, &m_pitch,
                              m_nCols * sizeof(T), m_nRows));
  CHECK_ERROR(
      cudaMemset2D(m_matrixValue, m_pitch, 0, m_nCols * sizeof(T), m_nRows));
}

template <>
inline MatrixGPUPitched<float> MatrixGPUPitched<float>::sgemm(
    const MatrixGPUPitched<float> &rhs) {
  int lhsNRows = (*this).m_nRows;
  int lhsNCols = (*this).m_nCols;
  int rhsNRows = rhs.m_nRows;
  int rhsNCols = rhs.m_nCols;
  ASSERT((*this).m_matrixValue != nullptr && rhs.m_matrixValue != nullptr,
         "matmul error: invalid matrix.\n")
  ASSERT(lhsNCols == rhsNRows, "matmul error: Matrix size mismatch.\n")
  MatrixGPUPitched<float> result(lhsNRows, rhsNCols);

  const int BM = 128;
  const int BN = 128;
  const int TM = 8;
  const int TN = 8;
  dim3 block_size(BM * BN / (TM * TN));
  dim3 grid_size((rhsNCols - 1) / BN + 1, (lhsNRows - 1) / BM + 1);
  sgemmKernel7<<<grid_size, block_size>>>(
      lhsNRows, rhsNRows, (*this).m_matrixValue, (*this).m_pitch,
      rhs.m_matrixValue, rhs.m_pitch, result.m_matrixValue, result.m_pitch);

  CHECK_ERROR(cudaGetLastError());
  CHECK_ERROR(cudaDeviceSynchronize());
  return result;
}

template <>
MatrixGPUPitched<half> MatrixGPUPitched<half>::hgemm(
    const MatrixGPUPitched<half> &rhs) {
  int lhsNRows = (*this).m_nRows;
  int lhsNCols = (*this).m_nCols;
  int rhsNRows = rhs.m_nRows;
  int rhsNCols = rhs.m_nCols;
  ASSERT((*this).m_matrixValue != nullptr && rhs.m_matrixValue != nullptr,
         "matmul error: invalid matrix.\n")
  ASSERT(lhsNCols == rhsNRows, "matmul error: Matrix size mismatch.\n")
  MatrixGPUPitched<half> result(lhsNRows, rhsNCols);
  const int BM = 128, BN = 256;
  dim3 block_size(256);
  dim3 grid_size((rhsNCols - 1) / BN + 1, (lhsNRows - 1) / BM + 1);
  hgemmKernel1<<<grid_size, block_size>>>(
      lhsNRows, rhsNCols, lhsNCols, (*this).m_matrixValue, (*this).m_pitch,
      rhs.m_matrixValue, rhs.m_pitch, result.m_matrixValue, result.m_pitch);
  CHECK_ERROR(cudaGetLastError());
  CHECK_ERROR(cudaDeviceSynchronize());
  return result;
}

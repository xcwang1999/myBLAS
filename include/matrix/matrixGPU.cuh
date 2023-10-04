#ifndef MYBLAS_INCLUDE_MATRIX_MATRIXGPU_CUH_
#define MYBLAS_INCLUDE_MATRIX_MATRIXGPU_CUH_

#include <cuda_fp16.h>

template <class T>
class MatrixGPU {
 public:
  MatrixGPU() noexcept;
  MatrixGPU(int nRows, int nCols) noexcept;
  MatrixGPU(int nRows, int nCols, T *matrixValue) noexcept;
  MatrixGPU(const MatrixGPU<T> &inputMatrix) noexcept;
  MatrixGPU(MatrixGPU<T> &&inputMatrix) noexcept;
  ~MatrixGPU() noexcept;

  MatrixGPU<T> &operator=(const MatrixGPU<T> &rhs) noexcept;
  MatrixGPU<T> &operator=(MatrixGPU<T> &&rhs) noexcept;

  void printMatrix();
  T *getMatrix();
  void ones(int nRows, int nCols);
  void zeros(int nRows, int nCols);
  MatrixGPU<T> matmul(const MatrixGPU<T> &rhs);
  MatrixGPU<float> sgemm(const MatrixGPU<float> &rhs);

  MatrixGPU<T> &transpose();

 private:
  int m_nRows;
  int m_nCols;
  int m_nElements;
  T *m_matrixValue;
};

extern template class MatrixGPU<float>;
extern template class MatrixGPU<half>;

template <class T>
class MatrixGPUPitched {
 public:
  MatrixGPUPitched() noexcept;
  MatrixGPUPitched(int nRows, int nCols) noexcept;
  MatrixGPUPitched(int nRows, int nCols, T *matrixValue) noexcept;
  MatrixGPUPitched(const MatrixGPUPitched<T> &inputMatrix) noexcept;
  MatrixGPUPitched(MatrixGPUPitched<T> &&inputMatrix) noexcept;
  ~MatrixGPUPitched() noexcept;

  MatrixGPUPitched<T> &operator=(const MatrixGPUPitched<T> &rhs) noexcept;
  MatrixGPUPitched<T> &operator=(MatrixGPUPitched<T> &&rhs) noexcept;

  void printMatrix();
  T *getMatrix();
  void zeros(int nRows, int nCols);
  MatrixGPUPitched<float> sgemm(const MatrixGPUPitched<float> &rhs);
  MatrixGPUPitched<half> hgemm(const MatrixGPUPitched<half> &rhs);

 private:
  int m_nRows;
  int m_nCols;
  int m_nElements;
  size_t m_pitch;
  T *m_matrixValue;
};

extern template class MatrixGPUPitched<float>;
extern template class MatrixGPUPitched<half>;

#endif  // MYBLAS_INCLUDE_MATRIX_MATRIXGPU_CUH_
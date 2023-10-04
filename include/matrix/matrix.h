#ifndef MYBLAS_INCLUDE_MATRIX_MATRIX_H_
#define MYBLAS_INCLUDE_MATRIX_MATRIX_H_

template <class T>
class Matrix {
 public:
  Matrix() noexcept;
  Matrix(int nRows, int nCols, const T *inputValue) noexcept;
  Matrix(const Matrix<T> &inputMatrix) noexcept;
  Matrix(Matrix<T> &&inputMatrix) noexcept;
  Matrix<T> &operator=(const Matrix<T> &rhs) noexcept;
  Matrix<T> &operator=(Matrix<T> &&rhs) noexcept;

  ~Matrix() noexcept;

  T getElement(int rowIndex, int colIndex);
  void setElement(int rowIndex, int colIndex, T elementValue);
  void reshape(int nRows, int nCols);
  int getNRows();
  int getNCols();
  int getNElements();
  void printValue();

  void zeros(int nRows, int nCols);
  void ones(int nRows, int nCols);
  void setMatrix(int nRows, int nCols, const T *inputValue);

  Matrix<T> matmul(const Matrix<T> &rhs);

  bool operator==(const Matrix<T> &rhs);

  template <typename U>
  friend Matrix<U> operator+(const Matrix<U> &lhs, const Matrix<U> &rhs);
  template <typename U>
  friend Matrix<U> operator+(const Matrix<U> &lhs, const U rhs);
  template <typename U>
  friend Matrix<U> operator+(const U lhs, const Matrix<U> &rhs);

  template <typename U>
  friend Matrix<U> operator-(const Matrix<U> &lhs, const Matrix<U> &rhs);
  template <typename U>
  friend Matrix<U> operator-(const Matrix<U> &lhs, const U rhs);
  template <typename U>
  friend Matrix<U> operator-(const U lhs, const Matrix<U> &rhs);

  template <typename U>
  friend Matrix<U> operator*(const Matrix<U> &lhs, const U rhs);
  template <typename U>
  friend Matrix<U> operator*(const U lhs, const Matrix<U> &rhs);

  template <typename U>
  friend Matrix<U> operator/(const Matrix<U> &lhs, const U rhs);
  template <typename U>
  friend Matrix<U> operator/(const U lhs, const Matrix<U> &rhs);

  template <typename U>
  friend Matrix<U> operator%(const Matrix<U> &lhs, const U rhs);
  template <typename U>
  friend Matrix<U> operator%(const U lhs, const Matrix<U> &rhs);

  template <typename U>
  friend float dot(const Matrix<U> &lhs, const Matrix<U> &rhs);

 private:
  int m_nRows;
  int m_nCols;
  int m_nElements;
  T *m_matrixValue;
};

extern template class Matrix<float>;
extern template class Matrix<double>;

#endif  // MYBLAS_INCLUDE_MATRIX_MATRIX_H_
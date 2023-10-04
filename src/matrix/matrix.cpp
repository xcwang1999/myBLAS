#include "matrix/matrix.h"

#include <iostream>

#include "utils.h"

template <class T>
Matrix<T>::Matrix() noexcept {
  m_nRows = 0;
  m_nCols = 0;
  m_nElements = 0;
  m_matrixValue = nullptr;
}

template <class T>
Matrix<T>::Matrix(int nRows, int nCols, const T *inputValue) noexcept {
  m_nRows = nRows;
  m_nCols = nCols;
  m_nElements = m_nRows * m_nCols;
  m_matrixValue = new T[m_nRows * m_nCols];
  memcpy(m_matrixValue, inputValue, m_nElements * sizeof(T));
}

template <class T>
Matrix<T>::Matrix(const Matrix<T> &inputMatrix) noexcept {
  if (inputMatrix.m_matrixValue != nullptr) {
    m_nRows = inputMatrix.m_nRows;
    m_nCols = inputMatrix.m_nCols;
    m_nElements = m_nRows * m_nCols;
    m_matrixValue = new T[m_nElements];
    memcpy(m_matrixValue, inputMatrix.m_matrixValue, m_nElements * sizeof(T));
  } else {
    m_nRows = 0;
    m_nCols = 0;
    m_nElements = 0;
    m_matrixValue = nullptr;
  }
}

template <class T>
Matrix<T>::Matrix(Matrix<T> &&inputMatrix) noexcept {
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
Matrix<T>::~Matrix() noexcept {
  if (m_matrixValue != nullptr) delete[] m_matrixValue;
  m_matrixValue = nullptr;
}

template <class T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &rhs) noexcept {
  if (this == &rhs) return *this;
  if (rhs.m_matrixValue == nullptr) {
    if (m_matrixValue != nullptr) {
      delete m_matrixValue;
      m_matrixValue = nullptr;
    }
  } else {
    m_nRows = rhs.m_nRows;
    m_nCols = rhs.m_nCols;
    m_nElements = m_nRows * m_nCols;
    if (m_matrixValue != nullptr) delete[] m_matrixValue;
    m_matrixValue = new T[m_nElements];
    memcpy(m_matrixValue, rhs.m_matrixValue, m_nElements * sizeof(T));
  }

  return *this;
}

template <class T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> &&rhs) noexcept {
  if (this == &rhs) return *this;
  if (rhs.m_matrixValue == nullptr) {
    if (m_matrixValue != nullptr) {
      delete m_matrixValue;
      m_matrixValue = nullptr;
    }
  } else {
    m_nRows = rhs.m_nRows;
    m_nCols = rhs.m_nCols;
    m_nElements = m_nRows * m_nCols;
    if (m_matrixValue != nullptr) delete[] m_matrixValue;
    m_matrixValue = rhs.m_matrixValue;
    rhs.m_matrixValue = nullptr;
  }

  return *this;
}

template <class T>
T Matrix<T>::getElement(int rowIndex, int colIndex) {
  if ((0 <= rowIndex) && (rowIndex < m_nRows) && (0 <= colIndex) &&
      (colIndex < m_nCols)) {
    return m_matrixValue[rowIndex * m_nCols + colIndex];
  } else if ((rowIndex < 0) && (-m_nRows <= rowIndex) && (colIndex < 0) &&
             (-m_nCols <= rowIndex)) {
    return m_matrixValue[(m_nRows + rowIndex) * m_nCols + (m_nCols + colIndex)];
  } else {
    std::cout << "getElement error: index out of range.\n";
    std::abort();
  }
}

template <class T>
void Matrix<T>::setElement(int rowIndex, int colIndex, T elementValue) {
  if ((0 <= rowIndex) && (rowIndex < m_nRows) && (0 <= colIndex) &&
      (colIndex < m_nCols)) {
    m_matrixValue[rowIndex * m_nCols + colIndex] = elementValue;
  } else if ((rowIndex < 0) && (-m_nRows <= rowIndex) && (colIndex < 0) &&
             (-m_nCols <= rowIndex)) {
    m_matrixValue[(m_nRows + rowIndex) * m_nCols + (m_nCols + colIndex)] =
        elementValue;
  } else {
    std::cout << "SetElement error: Index out of range.\n";
  }
}

template <class T>
void Matrix<T>::reshape(int nRows, int nCols) {
  ASSERT(nRows > 0 && nCols > 0, "reshape warning: No value in matrix.\n")
  ASSERT(nRows != -1 && nCols != -1,
         "reshape error: One of the axes must be given a value.\n")
  if (nRows * nCols == m_nElements) {
    m_nRows = nRows;
    m_nCols = nCols;
  } else if (nRows == -1 && nCols > 0 && (m_nElements % nCols) == 0) {
    m_nRows = m_nElements / nCols;
    m_nCols = nCols;
  } else if (nCols == -1 && nRows > 0 && (m_nElements % nRows) == 0) {
    m_nRows = nRows;
    m_nCols = m_nElements / nRows;
  } else {
    std::cout << "reshape error: Unable to reshape " << m_nRows << "x"
              << m_nCols << " matrix to " << nRows << "x" << nCols
              << " matrix.\n";
  }
}

template <class T>
int Matrix<T>::getNRows() {
  return m_nRows;
}

template <class T>
int Matrix<T>::getNCols() {
  return m_nCols;
}

template <class T>
int Matrix<T>::getNElements() {
  return m_nElements;
}

template <class T>
void Matrix<T>::zeros(int nRows, int nCols) {
  m_nRows = nRows;
  m_nCols = nCols;
  m_nElements = m_nRows * m_nCols;
  m_matrixValue = new T[m_nElements];
  memset(m_matrixValue, 0, m_nElements * sizeof(T));
}

template <class T>
void Matrix<T>::ones(int nRows, int nCols) {
  m_nRows = nRows;
  m_nCols = nCols;
  m_nElements = m_nRows * m_nCols;
  m_matrixValue = new T[m_nElements];
  for (int n = 0; n < m_nElements; n++) {
    m_matrixValue[n] = 1;
  }
}

template <class T>
void Matrix<T>::setMatrix(int nRows, int nCols, const T *inputValue) {
  m_nRows = nRows;
  m_nCols = nCols;
  m_nElements = m_nRows * m_nCols;
  m_matrixValue = new T[m_nRows * m_nCols];
  for (int n = 0; n < m_nElements; n++) {
    m_matrixValue[n] = inputValue[n];
  }
}

template <class T>
Matrix<T> Matrix<T>::matmul(const Matrix<T> &rhs) {
  int lhsNRows = (*this).m_nRows;
  int lhsNCols = (*this).m_nCols;
  int rhsNRows = rhs.m_nRows;
  int rhsNCols = rhs.m_nCols;
  ASSERT((*this).m_matrixValue != nullptr && rhs.m_matrixValue != nullptr,
         "matmul error: invalid matrix.\n")
  ASSERT(lhsNCols == rhsNRows, "matmul error: Matrix size mismatch.\n")
  Matrix<T> result;
  result.zeros(lhsNRows, rhsNCols);
  for (int i = 0; i < lhsNRows; i++) {
    for (int j = 0; j < rhsNCols; j++) {
      for (int k = 0; k < lhsNCols; k++) {
        result.m_matrixValue[i * rhsNCols + j] +=
            (*this).m_matrixValue[i * lhsNCols + k] *
            rhs.m_matrixValue[k * rhsNCols + j];
      }
    }
  }
  return result;
}

template <class T>
bool Matrix<T>::operator==(const Matrix<T> &rhs) {
  if ((*this).m_matrixValue != nullptr && rhs.m_matrixValue != nullptr) {
    if ((*this).m_nRows == rhs.m_nRows && (*this).m_nCols == rhs.m_nCols) {
      bool result = true;
      int lhsNElements = (*this).m_nElements;
      for (int n = 0; n < lhsNElements; n++) {
        if ((*this).m_matrixValue[n] != rhs.m_matrixValue[n]) {
          result = false;
          break;
        }
      }
      return result;
    } else {
      std::cout << "operator== warning: matrices have mismatched size.\n";
      return false;
    }
  } else {
    std::cout << "operator== warning: invalid matrix.\n";
    return false;
  }
}

template <class T>
void Matrix<T>::printValue() {
  ASSERT(m_matrixValue != nullptr, "printValue warning: No value to output.\n")
  std::cout << '[';
  for (int i = 0; i < m_nRows; i++) {
    if (i > 0) {
      std::cout << ' ';
    }
    for (int j = 0; j < m_nCols; j++) {
      std::cout << m_matrixValue[i * m_nCols + j];
      if (i == m_nRows - 1 && j == m_nCols - 1) {
        std::cout << "]\n";
      } else {
        std::cout << ", ";
      }
    }
    if (i < m_nRows - 1) {
      std::cout << "\n";
    }
  }
}

template <typename T>
Matrix<T> operator+(const Matrix<T> &lhs, const Matrix<T> &rhs) {
  ASSERT(lhs.m_matrixValue != nullptr && rhs.m_matrixValue != nullptr,
         "operator+ error: invalid matrix.\n")
  ASSERT(lhs.m_nRows == rhs.m_nRows && lhs.m_nCols == rhs.m_nCols,
         "operator+ error: matrices have mismatched size.\n")
  int lhsNElements = lhs.m_nElements;
  T *tempMatrix = new T[lhsNElements];
  for (int n = 0; n < lhsNElements; n++) {
    tempMatrix[n] = lhs.m_matrixValue[n] + rhs.m_matrixValue[n];
  }
  Matrix<T> result = Matrix<T>(lhs.m_nRows, lhs.m_nCols, tempMatrix);
  delete[] tempMatrix;
  return result;
}

template <typename T>
Matrix<T> operator+(const Matrix<T> &lhs, const T rhs) {
  ASSERT(rhs.m_matrixValue != nullptr, "operator+ error: invalid matrix.\n")
  int lhsNElements = lhs.m_nElements;
  for (int n = 0; n < lhsNElements; n++) {
    lhs.m_matrixValue[n] = lhs.m_matrixValue[n] + rhs;
  }
  return lhs;
}

template <typename T>
Matrix<T> operator+(const T lhs, const Matrix<T> &rhs) {
  ASSERT(rhs.m_matrixValue != nullptr, "operator+ error: invalid matrix.\n")
  int rhsNElements = rhs.m_nElements;
  for (int n = 0; n < rhsNElements; n++) {
    rhs.m_nElements[n] = lhs + rhs.m_matrixValue[n];
  }
  return rhs;
}

template <typename T>
Matrix<T> operator-(const Matrix<T> &lhs, const Matrix<T> &rhs) {
  ASSERT(lhs.m_matrixValue != nullptr && rhs.m_matrixValue != nullptr,
         "operator- error: invalid matrix.\n")
  ASSERT(lhs.m_nRows == rhs.m_nRows && lhs.m_nCols == rhs.m_nCols,
         "operator- error: matrices have mismatched size.\n")
  int lhsNElements = lhs.m_nElements;
  T *tempMatrix = new T[lhsNElements];
  for (int n = 0; n < lhsNElements; n++) {
    tempMatrix[n] = lhs.m_matrixValue[n] - rhs.m_matrixValue[n];
  }
  Matrix<T> result = Matrix<T>(lhs.m_nRows, lhs.m_nCols, tempMatrix);
  delete[] tempMatrix;
  return result;
}

template <typename T>
Matrix<T> operator-(const Matrix<T> &lhs, const T rhs) {
  ASSERT(rhs.m_matrixValue != nullptr, "operator- error: invalid matrix.\n")
  int lhsNElements = lhs.m_nElements;
  for (int n = 0; n < lhsNElements; n++) {
    lhs.m_matrixValue[n] = lhs.m_matrixValue[n] - rhs;
  }
  return lhs;
}

template <typename T>
Matrix<T> operator-(const T lhs, const Matrix<T> &rhs) {
  ASSERT(rhs.m_matrixValue != nullptr, "operator- error: invalid matrix.\n")
  int rhsNElements = rhs.m_nElements;
  for (int n = 0; n < rhsNElements; n++) {
    rhs.m_nElements[n] = lhs - rhs.m_matrixValue[n];
  }
  return rhs;
}

template <typename T>
Matrix<T> operator*(const Matrix<T> &lhs, const T rhs) {
  ASSERT(rhs.m_matrixValue != nullptr, "operator* error: invalid matrix.\n")
  int lhsNElements = lhs.m_nElements;
  for (int n = 0; n < lhsNElements; n++) {
    lhs.m_matrixValue[n] = lhs.m_matrixValue[n] * rhs;
  }
  return lhs;
}

template <typename T>
Matrix<T> operator*(const T lhs, const Matrix<T> &rhs) {
  ASSERT(rhs.m_matrixValue != nullptr, "operator* error: invalid matrix.\n")
  int rhsNElements = rhs.m_nElements;
  for (int n = 0; n < rhsNElements; n++) {
    rhs.m_nElements[n] = lhs * rhs.m_matrixValue[n];
  }
  return rhs;
}

template <typename T>
Matrix<T> operator/(const Matrix<T> &lhs, const T rhs) {
  ASSERT(rhs.m_matrixValue != nullptr, "operator/ error: invalid matrix.\n")
  int lhsNElements = lhs.m_nElements;
  for (int n = 0; n < lhsNElements; n++) {
    lhs.m_matrixValue[n] = lhs.m_matrixValue[n] / rhs;
  }
  return lhs;
}

template <typename T>
Matrix<T> operator/(const T lhs, const Matrix<T> &rhs) {
  ASSERT(rhs.m_matrixValue != nullptr, "operator/ error: invalid matrix.\n")
  int rhsNElements = rhs.m_nElements;
  for (int n = 0; n < rhsNElements; n++) {
    rhs.m_nElements[n] = lhs / rhs.m_matrixValue[n];
  }
  return rhs;
}

template <typename T>
Matrix<T> operator%(const Matrix<T> &lhs, const T rhs) {
  ASSERT(rhs.m_matrixValue != nullptr, "operator% error: invalid matrix.\n")
  int lhsNElements = lhs.m_nElements;
  for (int n = 0; n < lhsNElements; n++) {
    lhs.m_matrixValue[n] = lhs.m_matrixValue[n] % rhs;
  }
  return lhs;
}

template <typename T>
Matrix<T> operator%(const T lhs, const Matrix<T> &rhs) {
  ASSERT(rhs.m_matrixValue != nullptr, "operator% error: invalid matrix.\n")
  int rhsNElements = rhs.m_nElements;
  for (int n = 0; n < rhsNElements; n++) {
    rhs.m_nElements[n] = lhs % rhs.m_matrixValue[n];
  }
  return rhs;
}

template <typename T>
float dot(const Matrix<T> &lhs, const Matrix<T> &rhs) {
  ASSERT(lhs.m_matrixValue != nullptr && rhs.m_matrixValue != nullptr,
         "dot error: invalid matrix.\n")
  ASSERT(lhs.m_nRows == rhs.m_nRows && lhs.m_nCols == rhs.m_nCols,
         "dot error: matrices have mismatched size.\n")
  float result = 0.0;
  int lhsNElements = lhs.m_nElements;
  for (int n = 0; n < lhsNElements; n++) {
    result += lhs.m_matrixValue[n] * rhs.m_matrixValue[n];
  }
  return result;
}

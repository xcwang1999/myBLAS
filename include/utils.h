#ifndef MYBLAS_INCLUDE_UTILS_H_
#define MYBLAS_INCLUDE_UTILS_H_

#include <cuda_fp16.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

#define ASSERT(expr, message)                                   \
  if (!(expr)) {                                                \
    std::cerr << "Assertion failed: " << message << "\n"        \
              << "      File:       " << __FILE__ << "\n"       \
              << "      Line:       " << __LINE__ << std::endl; \
    std::abort();                                               \
  }

#define CHECK_ERROR(call)                                             \
  do {                                                                \
    const cudaError_t error_code = call;                              \
    if (error_code != cudaSuccess) {                                  \
      printf("CUDA Error:\n");                                        \
      printf("    File:       %s\n", __FILE__);                       \
      printf("    Line:       %d\n", __LINE__);                       \
      printf("    Error code: %d\n", error_code);                     \
      printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
      exit(1);                                                        \
    }                                                                 \
                                                                      \
  } while (0);

template <class T>
void printValue(const int nRows, const int nCols, const T *matrixValue) {
  ASSERT(matrixValue != nullptr, "printValue warning: No value to output.\n");
  std::cout << '[';
  for (int i = 0; i < nRows; i++) {
    if (i > 0) {
      std::cout << ' ';
    }
    for (int j = 0; j < nCols; j++) {
      std::cout << std::fixed << std::setprecision(4)
                << matrixValue[i * nCols + j];
      if (i == nRows - 1 && j == nCols - 1) {
        std::cout << "]\n";
      } else {
        std::cout << ", ";
      }
    }
    if (i < nRows - 1) {
      std::cout << "\n";
    }
  }
}

template <typename T>
void verifyResult(const T *resultA, const std::string &nameA, const T *resultB,
                  const std::string &nameB, const int length) {
  T sumOfAllError = 0.0;
  T maxError = 0.0;
  for (int n = 0; n < length; n++) {
    T error = abs(resultA[n] - resultB[n]);
    sumOfAllError += error;
    if (error > maxError) maxError = error;
  }
  std::cout << "sum of all error " << nameA << " vs " << nameB << " : "
            << sumOfAllError << "\n";
  std::cout << "maximum error " << nameA << " vs " << nameB << " : " << maxError
            << "\n";
}

inline float *halfToFloat(half *inputValue, int length) {
  float *tmp = new float[length];
  for (int n = 0; n < length; n++) {
    tmp[n] = __half2float(inputValue[n]);
  }
  return tmp;
}

#endif  // MYBLAS_INCLUDE_UTILS_H_

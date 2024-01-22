#pragma once

#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <filesystem>
#include <cmath>
#include "helper_macros.h"

namespace myblas {

template <typename T>
struct CudaDeleter {
  void operator()(T *ptr) { CHECK_CUDA(cudaFree(ptr)); }
};

template <typename T>
using AllocFuncType = cudaError_t (*)(T **devPtr, size_t size);

template <typename T>
using UniqueCudaPtr = std::unique_ptr<T[], CudaDeleter<T>>;

template <typename T, AllocFuncType<T> AllocFunc = cudaMalloc>
UniqueCudaPtr<T> make_unique_cuda(size_t size) {
  T *raw_ptr;
  CHECK_CUDA(AllocFunc(&raw_ptr, size * sizeof(T)));
  return UniqueCudaPtr<T>(raw_ptr);
}

template<typename T>
T make_random(const float min, const float max) {
  return min + static_cast<T>(rand()) / static_cast<T>(RAND_MAX / (max - min));
}

template<typename T>
MYBLAS_HOST_DEVICE bool verify(T const* src, T const* ref, int const length, const double allowedError=0.0){
  bool isAllEqual = true;
  for (int i=0; i<length; ++i){
    if (::std::abs(static_cast<double>(src[i] - ref[i])) > allowedError) {
      isAllEqual = false;
    }
  }

  return isAllEqual;
}

template<class T>
void print_matrix(const int nRows, const int nCols, const T* matrixValue, const ::std::string& label) {

  ::std::ofstream output;
  output.open((label + ".csv").c_str(), ::std::ios::out);
  for(int i=0; i<nRows; i++){
      if(i > 0) {output << "\n";}
      for(int j=0; j<nCols; j++){
          output << ::std::fixed << std::setprecision(3) << matrixValue[i*nCols+j];
          output <<  ", ";
      }
  }
  output.close();
}

inline float *halfToFloat(const half *inputValue, const int length) {
  float *tmp = new float[length];
  for (int n = 0; n < length; n++) {
    tmp[n] = __half2float(inputValue[n]);
  }
  return tmp;
}

template<>
inline void print_matrix<half>(const int nRows, const int nCols, const half* matrixValue, const ::std::string& label) {

  ::std::unique_ptr<float[]> matrixFloat{halfToFloat(matrixValue, nRows * nCols)};
  print_matrix<float>(nRows, nCols, matrixFloat.get(), label);
}

template<typename T>
MYBLAS_HOST_DEVICE void print_value(int const M, int const N, T const *matrixValue, const char *label){

  printf("matrix %s\n", label);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%.2f ", static_cast<float>(matrixValue[i * N + j]));
    }
    printf("\n");
  }
}

template <typename T>
MYBLAS_HOST_DEVICE void count_result(const T *resultA, const char* nameA, const T *resultB,
                  const char* nameB, const int length) {
  double sumOfAllError = 0.0;
  double maxError = 0.0;
  for (int n = 0; n < length; ++n) {
    double error = 0.0;
    error = ::std::abs(static_cast<double>(resultA[n] - resultB[n]));

    if (error > maxError) {
      maxError = error;
    }

    sumOfAllError += error;
  }

  printf("sum of all error %s vs %s : %f\n", nameA, nameB, sumOfAllError);
  printf("maximum error %s vs %s : %f\n", nameA, nameB, maxError);
}

} // namespace myblas
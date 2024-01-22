#pragma once

#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <filesystem>
#include <cmath>

namespace myblas {
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

#define ASSERT(expr, message)                                   \
  if (!(expr)) {                                                \
    std::cerr << "Assertion failed: " << message << "\n"        \
              << "      File:       " << __FILE__ << "\n"       \
              << "      Line:       " << __LINE__ << std::endl; \
    std::abort();                                               \
  }

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    const cudaError_t error_code = call;                              \
    if (error_code != cudaSuccess) {                                  \
      printf("CUDA Error:\n");                                        \
      printf("    File:       %s\n", __FILE__);                       \
      printf("    Line:       %d\n", __LINE__);                       \
      printf("    Error code: %d\n", error_code);                     \
      printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
      throw std::logic_error("cuda API failed");                      \
    }                                                                 \
                                                                      \
  } while (0);

#define CHECK_CUBLAS(call)                                               \
  do {                                                                   \
    const cublasStatus_t error_code = call;                              \
    if (error_code != CUBLAS_STATUS_SUCCESS) {                           \
      printf("CUDA Error:\n");                                           \
      printf("    File:       %s\n", __FILE__);                          \
      printf("    Line:       %d\n", __LINE__);                          \
      printf("    Error code: %d\n", error_code);                        \
      printf("    Error text: %s\n", cublasGetStatusString(error_code)); \
      throw std::logic_error("cuBLAS API failed");                       \
    }                                                                    \
                                                                         \
  } while (0);

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
bool verify(T const* src, T const* ref, int const length){
    bool isAllEqual = true;
    for (int i=0; i<length; ++i){
        if (src[i] != ref[i]){
            isAllEqual = false;
        }
    }

    return isAllEqual;
}

template<class T>
void print_matrix(const int nRows, const int nCols, const T* matrixValue, const ::std::string& label) {
    if(!matrixValue){
        ::std::cout << "printValue warning: No value to output.\n";
        return;
    }

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
    if(!matrixValue){
        ::std::cout << "printValue warning: No value to output.\n";
        return;
    }
    ::std::unique_ptr<float[]> matrixFloat{halfToFloat(matrixValue, nRows * nCols)};
    ::std::ofstream output;
    output.open((label + ".csv").c_str(), ::std::ios::out);
    for(int i=0; i<nRows; i++){
        if(i > 0) {output << "\n";}
        for(int j=0; j<nCols; j++){
            output << ::std::fixed << std::setprecision(3) << matrixFloat[i*nCols+j];
            output <<  ", ";
        }
    }
    output.close();
}

template <typename T>
void verifyResult(const T *resultA, const char* nameA, const T *resultB,
                  const char* nameB, const int length) {
  float sumOfAllError = 0.0;
  float maxError = 0.0;
  for (int n = 0; n < length; n++) {
    float error = abs(__half2float(resultA[n] - resultB[n]));
    sumOfAllError += error;
    if (error > maxError) maxError = error;
  }
  ::std::cout << "sum of all error " << nameA << " vs " << nameB << " : "
            << sumOfAllError << "\n";
  ::std::cout << "maximum error " << nameA << " vs " << nameB << " : " << maxError
            << "\n";
}

} // namespace myblas
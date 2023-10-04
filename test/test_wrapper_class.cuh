#ifndef GEMM_TEST_INCLUDE_TEST_TEST_WRAPPER_CLASS_CUH_
#define GEMM_TEST_INCLUDE_TEST_TEST_WRAPPER_CLASS_CUH_

#include <cblas.h>
#include <cublas_v2.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "matrix/matrixGPU.cuh"
#include "utils.h"
template <typename T, class Matrix, size_t size>
void testWrapper(int (&shapes)[size], int repeat, bool enableWriteFile = 0,
                 bool printMyMatrixRst = 0, bool printCublasRst = 0,
                 bool printopenblasRst = 0) {
  cudaEvent_t start, stop;
  float elapsed_time_mykernel;
  float elapsed_time_cublas;
  std::srand(std::time(nullptr));

  std::ofstream outfile;

  // Iterate over each shape
  for (int shapeid = 0; shapeid < size; shapeid++) {
    int M = shapes[shapeid];
    int K = shapes[shapeid];
    int N = shapes[shapeid];
    T *randAValue = new T[M * K];
    T *randBValue = new T[K * N];

    // Assign random values between 0 and 1 to A and B
    for (int n = 0; n < M * K; n++)
      randAValue[n] = static_cast<float>(rand()) / RAND_MAX;
    for (int n = 0; n < K * N; n++)
      randBValue[n] = static_cast<float>(rand()) / RAND_MAX;

    // Timer
    float myKerneltTime = 0.0;
    float cublasKernerltTime = 0.0;

    // Init my matrix
    std::cout << "M: " << M << "    "
              << "N: " << N << "    "
              << "K: " << K << "\n";

    Matrix myMat1GPU(M, K, randAValue);
    Matrix myMat2GPU(K, N, randBValue);
    Matrix myMat3GPU;

    // Repeat calculation
    for (int n = 0; n < repeat; n++) {
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);

      myMat3GPU = myMat1GPU.sgemm(myMat2GPU);

      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed_time_mykernel, start, stop);

      myKerneltTime += elapsed_time_mykernel / 1000;
    }
    myKerneltTime /= repeat;
    std::cout << "my kernel time: " << myKerneltTime << "\n";

    if (enableWriteFile) {
      outfile.open("execution_time.txt", std::ios::app);
      outfile << myKerneltTime << " ";
    }

    if (printMyMatrixRst) myMat3GPU.printMatrix();

    // Init cublas
    T *cublasRstHost = new T[M * N];
    memset(cublasRstHost, 0, M * N * sizeof(T));

    T *Adevice, *Bdevice, *cublasResDevice;
    CHECK_ERROR(cudaMalloc((void **)&Adevice, M * K * sizeof(T)));
    CHECK_ERROR(cudaMalloc((void **)&Bdevice, K * N * sizeof(T)));
    CHECK_ERROR(cudaMalloc((void **)&cublasResDevice, M * N * sizeof(T)));

    cublasSetVector(M * K, sizeof(T), randAValue, 1, Adevice, 1);
    cublasSetVector(K * N, sizeof(T), randBValue, 1, Bdevice, 1);
    cublasSetVector(M * N, sizeof(T), cublasRstHost, 1, cublasResDevice, 1);

    // repeat calculation
    for (int n = 0; n < repeat; n++) {
      cublasHandle_t handle;
      cublasCreate(&handle);
      const float alpha = 1;
      const float beta = 0;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);

      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, Bdevice, N,
                  Adevice, K, &beta, cublasResDevice, N);

      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed_time_cublas, start, stop);
      cublasDestroy(handle);
      cublasKernerltTime += elapsed_time_cublas / 1000;
    }
    cublasKernerltTime /= repeat;
    std::cout << "cublas kernel time: " << cublasKernerltTime << "\n";

    if (enableWriteFile) {
      outfile << cublasKernerltTime << "\n";
      outfile.close();
    }

    // calculate error between my kernel and cublas
    cublasGetVector(M * N, sizeof(T), cublasResDevice, 1, cublasRstHost, 1);
    T *mykernelRstHost = myMat3GPU.getMatrix();

    // init openblas
    T *openblasRst = new T[M * N];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0,
                randAValue, K, randBValue, N, 0.0, openblasRst, N);

    // verify result

    verifyResult<T>(mykernelRstHost, "mekernel", cublasRstHost, "cublas",
                    M * N);
    verifyResult<T>(mykernelRstHost, "mekernel", openblasRst, "openblas",
                    M * N);
    verifyResult<T>(cublasRstHost, "cublas", openblasRst, "openblas", M * N);
    std::cout << "-----------------------------------"
              << "\n";

    if (printCublasRst) printValue(M, N, cublasRstHost);
    if (printopenblasRst) printValue(M, N, openblasRst);

    delete[] cublasRstHost;
    delete[] openblasRst;
    delete[] mykernelRstHost;

    CHECK_ERROR(cudaFree(Adevice));
    CHECK_ERROR(cudaFree(Bdevice));
    CHECK_ERROR(cudaFree(cublasResDevice));

    delete[] randAValue;
    delete[] randBValue;
  }
}

#endif  // GEMM_TEST_INCLUDE_TEST_TEST_WRAPPER_CLASS_CUH_
#ifndef GEMM_TEST_INCLUDE_TEST_TEST_HGEMM_PADDING_CUH_
#define GEMM_TEST_INCLUDE_TEST_TEST_HGEMM_PADDING_CUH_

#include <cblas.h>
#include <cublas_v2.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "hgemm/kernels.cuh"
#include "utils.h"

template <size_t size>
void testHgemmPadding(int (&shapes)[size], int repeat, bool enableWriteFile = 0,
                      bool printMyKernelRst = 0, bool printCublasRst = 0) {
  typedef half T;
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
    T *randAValue = (T *)malloc(M * K * sizeof(T));
    T *randBValue = (T *)malloc(K * N * sizeof(T));

    // Assign random values between 0 and 1 to A and B
    for (int n = 0; n < M * K; n++)
      randAValue[n] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    for (int n = 0; n < K * N; n++)
      randBValue[n] = __float2half(static_cast<float>(rand()) / RAND_MAX);

    // Timer
    float myKerneltTime = 0.0;
    float cublasKerneltTime = 0.0;

    // Init my matrix
    std::cout << "shape: " << shapes[shapeid] << "\n";

    T *Adevice;
    T *Bdevice;
    T *Cdevice;
    size_t Apitch;
    size_t Bpitch;
    size_t Cpitch;
    CHECK_ERROR(cudaMallocPitch((void **)&Adevice, &Apitch, K * sizeof(T), M));
    CHECK_ERROR(cudaMallocPitch((void **)&Bdevice, &Bpitch, N * sizeof(T), K));
    CHECK_ERROR(cudaMallocPitch((void **)&Cdevice, &Cpitch, N * sizeof(T), M));

    std::cout << "M: " << M << "    "
              << "N: " << N << "    "
              << "K: " << K << "\n";
    std::cout << "Apitch: " << Apitch << "    "
              << "Bptich:" << Bpitch << "    "
              << "Cpitch:" << Cpitch << "\n";
    CHECK_ERROR(cudaMemset2D(Adevice, Apitch, 0, K * sizeof(T), M));
    CHECK_ERROR(cudaMemset2D(Bdevice, Bpitch, 0, N * sizeof(T), K));
    CHECK_ERROR(cudaMemset2D(Cdevice, Cpitch, 0, N * sizeof(T), M));
    CHECK_ERROR(cudaMemcpy2D(Adevice, Apitch, randAValue, K * sizeof(T),
                             K * sizeof(T), M, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy2D(Bdevice, Bpitch, randBValue, N * sizeof(T),
                             N * sizeof(T), K, cudaMemcpyHostToDevice));

    const int BM = 128, BN = 256;
    dim3 blockDim(256);
    dim3 gridDim((N - 1) / BN + 1, (M - 1) / BM + 1);
    // Repeat calculation
    for (int n = 0; n < repeat; n++) {
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);

      hgemmKernel1<<<gridDim, blockDim>>>(M, N, K, Adevice, Apitch, Bdevice,
                                          Bpitch, Cdevice, Cpitch);
      CHECK_ERROR(cudaGetLastError());
      CHECK_ERROR(cudaDeviceSynchronize());
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

    T *mykernelRstHost = (T *)malloc(M * N * sizeof(T));
    CHECK_ERROR(cudaMemcpy2D(mykernelRstHost, N * sizeof(T), Cdevice, Cpitch,
                             N * sizeof(T), M, cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(Adevice));
    CHECK_ERROR(cudaFree(Bdevice));
    CHECK_ERROR(cudaFree(Cdevice));

    // Init cublas
    T *cublasRstHost = (T *)malloc(M * N * sizeof(T));
    memset(cublasRstHost, 0, M * N * sizeof(T));

    T *cublasA;
    T *cublasB;
    T *cublasRstDevice;
    CHECK_ERROR(cudaMalloc((void **)&cublasA, M * K * sizeof(T)));
    CHECK_ERROR(cudaMalloc((void **)&cublasB, K * N * sizeof(T)));
    CHECK_ERROR(cudaMalloc((void **)&cublasRstDevice, M * N * sizeof(T)));

    cublasSetVector(M * K, sizeof(T), randAValue, 1, cublasA, 1);
    cublasSetVector(K * N, sizeof(T), randBValue, 1, cublasB, 1);
    cublasSetVector(M * N, sizeof(T), cublasRstHost, 1, cublasRstDevice, 1);

    // repeat calculation
    for (int n = 0; n < repeat; n++) {
      cublasHandle_t handle;
      cublasCreate(&handle);
      const half alpha = 1;
      const half beta = 0;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);

      cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, cublasB, N,
                  cublasA, K, &beta, cublasRstDevice, N);

      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed_time_cublas, start, stop);
      cublasDestroy(handle);
      cublasKerneltTime += elapsed_time_cublas / 1000;
    }
    cublasKerneltTime /= repeat;
    std::cout << "cublas kernel time: " << cublasKerneltTime << "\n";

    if (enableWriteFile) {
      outfile << cublasKerneltTime << "\n";
      outfile.close();
    }

    // calculate error between my kernel and cublas
    cublasGetVector(M * N, sizeof(T), cublasRstDevice, 1, cublasRstHost, 1);

    // convert result matrix from half to float
    float *myKernelRstFloat = halfToFloat(mykernelRstHost, M * N);
    float *cublasRstFloat = halfToFloat(cublasRstHost, M * N);

    verifyResult<float>(myKernelRstFloat, "mykernel", cublasRstFloat, "cublas",
                        M * N);
    std::cout << "-----------------------------------"
              << "\n";

    if (printMyKernelRst) printValue<float>(M, N, myKernelRstFloat);
    if (printCublasRst) printValue<float>(M, N, cublasRstFloat);

    CHECK_ERROR(cudaFree(cublasA));
    CHECK_ERROR(cudaFree(cublasB));
    CHECK_ERROR(cudaFree(cublasRstDevice));

    free(cublasRstHost);
    free(mykernelRstHost);

    free(randAValue);
    free(randBValue);

    delete[] myKernelRstFloat;
    delete[] cublasRstFloat;
  }
}

#endif  // GEMM_TEST_INCLUDE_TEST_TEST_HGEMM_PADDING_CUH_
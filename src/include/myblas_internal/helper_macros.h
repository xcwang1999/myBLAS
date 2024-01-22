#pragma once

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define MYBLAS_HOST_DEVICE __forceinline__ __device__ __host__
#define MYBLAS_DEVICE __forceinline__ __device__
#elif defined(__CUDACC_RTC__)
#define MYBLAS_HOST_DEVICE __forceinline__ __device__
#define MYBLAS_DEVICE __forceinline__ __device__
#else
#define MYBLAS_HOST_DEVICE inline
#define MYBLAS_DEVICE inline
#endif

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

namespace myblas {

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

#define LDMATRIX_X1(reg, smem_addr)                                     \
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" \
               : "=r"(dst_reg)                                          \
               : "r"(smem_addr));

#define LDMATRIX_X2(reg0, reg1, smem_addr)                                  \
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
               : "=r"(reg0), "=r"(reg1)                                     \
               : "r"(smem_addr));

#define LDMATRIX_X4(reg0, reg1, reg2, reg3, smem_addr)                     \
  asm volatile(                                                            \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
      : "=r"(reg0), "=r"(reg1), "=r"(reg2), "=r"(reg3)                     \
      : "r"(smem_addr));

#define HMMA16816(D0, D1, A0, A1, A2, A3, B0, B1, C0, C1)                     \
  asm volatile(                                                               \
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, " \
      "%4, %5}, {%6, %7}, {%8, %9};\n"                                        \
      : "=r"(D0), "=r"(D1)                                                    \
      : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), "r"(C0),        \
        "r"(C1));

}  // namespace myblas

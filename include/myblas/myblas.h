#pragma once

#include "sgemm/kernels.h"
#include "transpose/kernels.h"

namespace myblas::sgemm {

struct KernelId1 {
  static constexpr char const label[] = "sgemmV1";
};
struct KernelId2 {
  static constexpr char const label[] = "sgemmV2";
};
struct KernelId3 {
  static constexpr char const label[] = "sgemmV3";
};
struct KernelId4 {
  static constexpr char const label[] = "sgemmV4";
};
struct KernelId5 {
  static constexpr char const label[] = "sgemmV5";
};
struct KernelId6 {
  static constexpr char const label[] = "sgemmV6";
};

template <typename KernelId>
void sgemm(const int M, const int N, const int K, float* A, float* B, float* C);

extern template void sgemm<KernelId1>(const int M, const int N, const int K,
                                      float* A, float* B, float* C);
extern template void sgemm<KernelId2>(const int M, const int N, const int K,
                                      float* A, float* B, float* C);
extern template void sgemm<KernelId3>(const int M, const int N, const int K,
                                      float* A, float* B, float* C);
extern template void sgemm<KernelId4>(const int M, const int N, const int K,
                                      float* A, float* B, float* C);
extern template void sgemm<KernelId5>(const int M, const int N, const int K,
                                      float* A, float* B, float* C);
extern template void sgemm<KernelId6>(const int M, const int N, const int K,
                                      float* A, float* B, float* C);

}  // namespace myblas::sgemm
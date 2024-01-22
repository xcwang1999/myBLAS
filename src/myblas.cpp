#include "myblas/myblas.h"

#include <cuda_runtime.h>

namespace myblas {

namespace sgemm {
template <typename KernelId>
void sgemm(const int M, const int N, const int K, float* A, float* B,
           float* C) {
  if constexpr (std::is_same_v<KernelId, KernelId1>) {
    sgemmV1(M, N, K, A, B, C);
  } else if constexpr (std::is_same_v<KernelId, KernelId2>) {
    sgemmV2(M, N, K, A, B, C);
  } else if constexpr (std::is_same_v<KernelId, KernelId3>) {
    sgemmV3(M, N, K, A, B, C);
  } else if constexpr (std::is_same_v<KernelId, KernelId4>) {
    sgemmV4(M, N, K, A, B, C);
  } else if constexpr (std::is_same_v<KernelId, KernelId5>) {
    sgemmV5(M, N, K, A, B, C);
  } else if constexpr (std::is_same_v<KernelId, KernelId6>) {
    sgemmV6(M, N, K, A, B, C);
  }
}

template void sgemm<KernelId1>(const int M, const int N, const int K, float* A,
                             float* B, float* C);
template void sgemm<KernelId2>(const int M, const int N, const int K, float* A,
                             float* B, float* C);
template void sgemm<KernelId3>(const int M, const int N, const int K, float* A,
                             float* B, float* C);
template void sgemm<KernelId4>(const int M, const int N, const int K, float* A,
                             float* B, float* C);
template void sgemm<KernelId5>(const int M, const int N, const int K, float* A,
                             float* B, float* C);
template void sgemm<KernelId6>(const int M, const int N, const int K, float* A,
                             float* B, float* C);
}  // namespace sgemm

} // namespace myblas
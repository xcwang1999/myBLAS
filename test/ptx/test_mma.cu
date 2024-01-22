#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include "myblas_internal/helper_macros.h"
#include "myblas_internal/layout.h"
#include "myblas_internal/utils.h"

using namespace myblas;

namespace {

__device__ bool passed_device = false;

MYBLAS_DEVICE void init_data(half* data, const int M, const int N,
                             const int ld) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      data[(j + i * ld)] = half{(j + i * ld) / 16};
    }
  }
}

MYBLAS_DEVICE void ref_matmul(const int M, const int N, const int K,
                              const half* __restrict__ A,
                              const half* __restrict__ B,
                              half* __restrict__ C) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      half sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += A[m * K + k] * B[n * K + k];
      }
      C[m * N + n] = sum;
    }
  }
}

}  // namespace

__global__ void test_mma() {
  constexpr int const M = 16;
  constexpr int const N = 8;
  constexpr int const K = 16;
  __shared__ half A[M * K];
  __shared__ half B[N * K];
  __shared__ half D[M * N];

  if (threadIdx.x == 0) {
    init_data(A, M, K, K);
    init_data(B, N, K, K);
  }

  __syncthreads();

  void* A_smem_addr;
  void* B_smem_addr;

  uint32_t A_reg[4];
  uint32_t B_reg[2];
  uint32_t C_reg[2];
  uint32_t D_reg[2];

  memset(&C_reg[0], 0, sizeof(uint32_t));
  memset(&C_reg[1], 0, sizeof(uint32_t));

  {
    constexpr int thread_group_m = 2;
    constexpr int thread_group_n = 2;
    constexpr int thread_local_m = 8;
    constexpr int thread_local_n = 1;
    using ThreadShape = Shape<Shape<Int<thread_local_m>, Int<thread_local_n>>,
                              Shape<Int<thread_group_m>, Int<thread_group_n>>>;
    using ThreadStride =
        Stride<Stride<Int<1>, Int<0>>, Stride<Int<8>, Int<16>>>;
    auto const thread_index = reshape<ThreadShape, ThreadStride>(threadIdx.x);

    using LoadStride =
        Stride<Stride<Int<16>, Int<0>>, Stride<Int<128>, Int<8>>>;
    A_smem_addr = A + get_linear_idx<LoadStride>(thread_index);

    uint32_t smem_space_addr = __cvta_generic_to_shared(A_smem_addr);

    LDMATRIX_X4(A_reg[0], A_reg[1], A_reg[2], A_reg[3], smem_space_addr)
  }

  {
    if (threadIdx.x < 16) {
      constexpr int thread_local_m = 8;
      constexpr int thread_local_n = 2;
      using ThreadShape = Shape<Int<thread_local_m>, Int<thread_local_n>>;
      using ThreadStride = Stride<Int<1>, Int<8>>;

      auto const thread_index = reshape<ThreadShape, ThreadStride>(threadIdx.x);

      using LoadStride = Stride<Int<16>, Int<8>>;
      B_smem_addr = B + get_linear_idx<LoadStride>(thread_index);
    } else {
      B_smem_addr = B;
    }

    uint32_t smem_space_addr = __cvta_generic_to_shared(B_smem_addr);

    LDMATRIX_X2(B_reg[0], B_reg[1], smem_space_addr)
  }

  HMMA16816(D_reg[0], D_reg[1], A_reg[0], A_reg[1], A_reg[2], A_reg[3],
            B_reg[0], B_reg[1], C_reg[0], C_reg[1])

  __syncthreads();

  constexpr int reg_m = 2;

  using RegisterShape = Shape<Int<reg_m>>;
  using RegisterStride = Stride<Int<1>>;

#pragma unroll
  for (int reg_linear_idx = 0; reg_linear_idx < reg_m; reg_linear_idx++) {
    constexpr int thread_local_m = 8;
    constexpr int thread_local_n = 4;
    using ThreadShape = Shape<Int<thread_local_m>, Int<thread_local_n>>;
    using ThreadStride = Stride<Int<4>, Int<1>>;
    auto const thread_index = reshape<ThreadShape, ThreadStride>(threadIdx.x);

    auto const reg_index =
        reshape<RegisterShape, RegisterStride>(reg_linear_idx);
    auto const store_index = make_index(thread_index, reg_index);

    using StoreStride = Stride<Stride<Int<8>, Int<2>>, Stride<Int<64>>>;

    int const smem_out_linear_idx = get_linear_idx<StoreStride>(store_index);

#pragma unroll
    for (int n = 0; n < 2; ++n) {
      D[smem_out_linear_idx + n] =
          reinterpret_cast<half*>(&D_reg[reg_linear_idx])[n];
    }
  }

  __syncthreads();

  if (threadIdx.x > 0) return;

  half D_ref[M * N];
  ref_matmul(M, N, K, A, B, D_ref);
  passed_device = verify(D, D_ref, M * N);

  if (!passed_device) {
    print_value(M, N, D_ref, "D_ref");
    print_value(M, N, D, "D");
    count_result(D, "D", D_ref, "D_ref", M * N);
  }
}

TEST(test_mma, HMMA16816) {
  test_mma<<<1, 32>>>();
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
  bool passed_host = false;
  CHECK_CUDA(cudaMemcpyFromSymbol(&passed_host, passed_device, sizeof(bool), 0,
                                  cudaMemcpyDeviceToHost))
  ASSERT_TRUE(passed_host);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

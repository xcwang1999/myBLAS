#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include "myblas_internal/helper_macros.h"
#include "myblas_internal/layout.h"
#include "myblas_internal/utils.h"

using namespace myblas;

namespace {

__device__ bool passed_device_x1 = false;
__device__ bool passed_device_x2 = false;
__device__ bool passed_device_x4 = false;

MYBLAS_DEVICE void init_data(half *data, const int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = i;
  }
}

}  // namespace

__global__ void test_ldmatrix_x1() {
  constexpr int const M = 8;
  constexpr int const N = 8;
  __shared__ half data[M * N];

  if (threadIdx.x == 0) {
    init_data(data, M * N);
  }

  __syncthreads();

  void *smem_addr;
  if (threadIdx.x < 8) {
    constexpr int thread_local_m = 8;
    using ThreadShape = Shape<Int<thread_local_m>>;
    using ThreadStride = Stride<Int<1>>;
    auto const thread_index = reshape<ThreadShape, ThreadStride>(threadIdx.x);

    using LoadStride = Stride<Int<8>>;
    smem_addr = data + get_linear_idx<LoadStride>(thread_index);
  } else {
    smem_addr = data;
  }

  uint32_t smem_space_addr = __cvta_generic_to_shared(smem_addr);
  uint32_t dst_reg;

  LDMATRIX_X1(dst_reg, smem_space_addr)

  __shared__ half data_out[M * N];

  constexpr int thread_local_m = 8;
  constexpr int thread_local_n = 4;
  using ThreadShape = Shape<Int<thread_local_m>, Int<thread_local_n>>;
  using ThreadStride = Stride<Int<4>, Int<1>>;
  auto const index = reshape<ThreadShape, ThreadStride>(threadIdx.x);

  using StoreStrideType = Stride<Int<8>, Int<2>>;

  int const smem_out_linear_idx = get_linear_idx<StoreStrideType>(index);

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    data_out[smem_out_linear_idx + i] = reinterpret_cast<half *>(&dst_reg)[i];
  }

  __syncthreads();

  if (threadIdx.x > 0) return;

  passed_device_x1 = verify(data_out, data, M * N);

  if (!passed_device_x1) {
    print_value(M, N, data, "data");
    print_value(M, N, data_out, "out");
  }
}

__global__ void test_ldmatrix_x2() {
  constexpr int const N = 8;
  constexpr int const K = 16;
  __shared__ half data[N * K];

  if (threadIdx.x == 0) {
    init_data(data, N * K);
  }

  __syncthreads();

  void *smem_addr;

  if (threadIdx.x < 16) {
    constexpr int thread_local_m = 8;
    constexpr int thread_local_n = 2;
    using ThreadShape = Shape<Int<thread_local_m>, Int<thread_local_n>>;
    using ThreadStride = Stride<Int<1>, Int<8>>;

    auto const thread_index = reshape<ThreadShape, ThreadStride>(threadIdx.x);

    using LoadStride = Stride<Int<16>, Int<8>>;
    smem_addr = data + get_linear_idx<LoadStride>(thread_index);
  } else {
    smem_addr = data;
  }

  uint32_t smem_space_addr = __cvta_generic_to_shared(smem_addr);
  uint32_t dst_reg[2];

  LDMATRIX_X2(dst_reg[0], dst_reg[1], smem_space_addr)

  __shared__ half data_out[N * K];

#pragma unroll
  for (int reg_linear_idx = 0; reg_linear_idx < 2; reg_linear_idx++) {
    constexpr int thread_local_m = 8;
    constexpr int thread_local_n = 4;
    using ThreadShape = Shape<Int<thread_local_m>, Int<thread_local_n>>;
    using ThreadStride = Stride<Int<4>, Int<1>>;
    auto const thread_index = reshape<ThreadShape, ThreadStride>(threadIdx.x);
    auto const store_index =
        make_index(thread_index, make_index(DynamicInt{reg_linear_idx}));
    using StoreStride = Stride<Stride<Int<16>, Int<2>>, Stride<Int<8>>>;

    int const smem_out_linear_idx = get_linear_idx<StoreStride>(store_index);

#pragma unroll
    for (int n = 0; n < 2; ++n) {
      data_out[smem_out_linear_idx + n] =
          reinterpret_cast<half *>(&dst_reg[reg_linear_idx])[n];
    }
  }
  __syncthreads();

  if (threadIdx.x > 0) return;

  passed_device_x2 = verify(data_out, data, N * K);

  if (!passed_device_x2) {
    print_value(N, K, data, "data");
    print_value(N, K, data_out, "out");
  }
}

__global__ void test_ldmatrix_x4() {
  constexpr int const M = 16;
  constexpr int const K = 16;
  __shared__ half data[M * K];

  if (threadIdx.x == 0) {
    init_data(data, M * K);
  }
  __syncthreads();

  void *smem_addr;

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
    smem_addr = data + get_linear_idx<LoadStride>(thread_index);
  }

  uint32_t smem_space_addr = __cvta_generic_to_shared(smem_addr);
  uint32_t dst_reg[4];

  LDMATRIX_X4(dst_reg[0], dst_reg[1], dst_reg[2], dst_reg[3], smem_space_addr)

  __shared__ half data_out[M * K];

  constexpr int reg_m = 2;
  constexpr int reg_n = 2;

  constexpr int reg_num = reg_m * reg_n;

  using RegisterShape = Shape<Int<reg_m>, Int<reg_n>>;
  using RegisterStride = Stride<Int<1>, Int<2>>;

#pragma unroll
  for (int reg_linear_idx = 0; reg_linear_idx < reg_num; reg_linear_idx++) {
    constexpr int thread_local_m = 8;
    constexpr int thread_local_n = 4;
    using ThreadShape = Shape<Int<thread_local_m>, Int<thread_local_n>>;
    using ThreadStride = Stride<Int<4>, Int<1>>;
    auto const thread_index = reshape<ThreadShape, ThreadStride>(threadIdx.x);

    auto const reg_index =
        reshape<RegisterShape, RegisterStride>(reg_linear_idx);
    auto const store_index = make_index(thread_index, reg_index);

    using StoreStride =
        Stride<Stride<Int<16>, Int<2>>, Stride<Int<128>, Int<8>>>;

    int const smem_out_linear_idx = get_linear_idx<StoreStride>(store_index);

#pragma unroll
    for (int n = 0; n < 2; ++n) {
      data_out[smem_out_linear_idx + n] =
          reinterpret_cast<half *>(&dst_reg[reg_linear_idx])[n];
    }
  }

  __syncthreads();

  if (threadIdx.x > 0) return;

  passed_device_x4 = verify(data_out, data, M * K);

  if (!passed_device_x4) {
    print_value(M, K, data, "data");
    print_value(M, K, data_out, "out");
  }
}

TEST(test_ldmatrix, x1) {
  test_ldmatrix_x1<<<1, 32>>>();
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
  bool passed_host = false;
  CHECK_CUDA(cudaMemcpyFromSymbol(&passed_host, passed_device_x1, sizeof(bool),
                                  0, cudaMemcpyDeviceToHost))
  ASSERT_TRUE(passed_host);
}

TEST(test_ldmatrix, x2) {
  test_ldmatrix_x2<<<1, 32>>>();
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
  bool passed_host = false;
  CHECK_CUDA(cudaMemcpyFromSymbol(&passed_host, passed_device_x2, sizeof(bool),
                                  0, cudaMemcpyDeviceToHost))
  ASSERT_TRUE(passed_host);
}

TEST(test_ldmatrix, x4) {
  test_ldmatrix_x4<<<1, 32>>>();
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
  bool passed_host = false;
  CHECK_CUDA(cudaMemcpyFromSymbol(&passed_host, passed_device_x4, sizeof(bool),
                                  0, cudaMemcpyDeviceToHost))
  ASSERT_TRUE(passed_host);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
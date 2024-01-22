#include <gtest/gtest.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "myblas/myblas.h"
#include "myblas_internal/utils.h"

namespace {

using Element = float;

template<typename KernelId>
class TestSgemmKernel : public ::testing::Test {
 protected:
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int M = 1024 * 1;
  int N = 1024 * 2;
  int K = 1024 * 3;
  cublasHandle_t cublasHandle;
  std::unique_ptr<Element[]> A_host, B_host, C_host;
  std::unique_ptr<Element[]> C_cublas_result, C_mykernel_result;
  myblas::UniqueCudaPtr<Element> A_device, B_device, C_device;
  int A_size, B_size, C_size;

  void ResetDevice();
  void SetValue();
  void SetUp() override;
  void TearDown() override;
  void TestRoutine();
};

template<typename KernelId>
void TestSgemmKernel<KernelId>::ResetDevice() {
  CHECK_CUDA(cudaMemcpy(A_device.get(), A_host.get(), A_size * sizeof(Element),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_device.get(), B_host.get(), B_size * sizeof(Element),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(C_device.get(), C_host.get(), C_size * sizeof(Element),
                        cudaMemcpyHostToDevice));
}

template<typename KernelId>
void TestSgemmKernel<KernelId>::SetValue() {
  for (int i = 0; i < A_size; ++i)
    A_host[i] = myblas::make_random<Element>(0, 1);
  for (int i = 0; i < B_size; ++i)
    B_host[i] = myblas::make_random<Element>(0, 1);

  memset(C_host.get(), 0, C_size * sizeof(Element));
}

template<typename KernelId>
void TestSgemmKernel<KernelId>::SetUp() {
  A_size = M * K;
  B_size = K * N;
  C_size = M * N;
  A_host = std::make_unique<Element[]>(A_size);
  B_host = std::make_unique<Element[]>(B_size);
  C_host = std::make_unique<Element[]>(C_size);
  A_device = myblas::make_unique_cuda<Element, cudaMalloc>(A_size);
  B_device = myblas::make_unique_cuda<Element, cudaMalloc>(B_size);
  C_device = myblas::make_unique_cuda<Element, cudaMalloc>(C_size);

  C_mykernel_result = std::make_unique<Element[]>(C_size);
  C_cublas_result = std::make_unique<Element[]>(C_size);

  SetValue();

  CHECK_CUBLAS(cublasCreate(&cublasHandle));
}

template<typename KernelId>
void TestSgemmKernel<KernelId>::TearDown() {
  CHECK_CUBLAS(cublasDestroy(cublasHandle));
  A_device.reset(nullptr);
  B_device.reset(nullptr);
  C_device.reset(nullptr);
}

template<typename KernelId>
void TestSgemmKernel<KernelId>::TestRoutine() {
  ResetDevice();
  CHECK_CUDA(cudaDeviceSynchronize());

  myblas::sgemm::sgemm<KernelId>(M, N, K, A_device.get(), B_device.get(),
                                 C_device.get());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(C_mykernel_result.get(), C_device.get(),
                        C_size * sizeof(Element), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaDeviceSynchronize());

  ResetDevice();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                           &alpha, B_device.get(), N, A_device.get(), K, &beta,
                           C_device.get(), N));
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(C_cublas_result.get(), C_device.get(),
                        C_size * sizeof(Element), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaDeviceSynchronize());

  const bool isAllEqual =
      myblas::verify(C_mykernel_result.get(), C_cublas_result.get(), C_size);

  EXPECT_TRUE(isAllEqual);

  if (!isAllEqual) {
    myblas::print_matrix(M, N, C_mykernel_result.get(), KernelId::label);
    myblas::print_matrix(M, N, C_cublas_result.get(), "cublasVS" + std::string(KernelId::label));
  }
}

using KernelTypes = ::testing::Types<
    myblas::sgemm::KernelId1,
    myblas::sgemm::KernelId2,
    myblas::sgemm::KernelId3,
    myblas::sgemm::KernelId4,
    myblas::sgemm::KernelId5,
    myblas::sgemm::KernelId6
    >;

TYPED_TEST_SUITE(TestSgemmKernel, KernelTypes);

TYPED_TEST(TestSgemmKernel, TestSgemm) {
  this->TestRoutine();
}
}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


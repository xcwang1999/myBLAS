#include <benchmark/benchmark.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "myblas/myblas.h"
#include "myblas_internal/helper_macros.h"
#include "myblas_internal/utils.h"

namespace {

using Element = float;

class BenchmarkSgemm : public benchmark::Fixture {
 protected:
  cublasHandle_t cublasHandle;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int M, N, K;
  double FLOPS;
  double TFLOPS;
  std::unique_ptr<Element[]> A_host, B_host, C_host;
  myblas::UniqueCudaPtr<Element> A_device, B_device, C_device;
  int A_size, B_size, C_size;

  void ResetDevice();

  void GetTflops();

  void SetHostValue();

  void SetUp(const ::benchmark::State& state) override;

  void TearDown(const ::benchmark::State& state) override;
};

void BenchmarkSgemm::ResetDevice() {
  CHECK_CUDA(cudaMemcpy(A_device.get(), A_host.get(), A_size * sizeof(Element),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_device.get(), B_host.get(), B_size * sizeof(Element),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(C_device.get(), C_host.get(), C_size * sizeof(Element),
                        cudaMemcpyHostToDevice));
}

void BenchmarkSgemm::GetTflops() {
  FLOPS = 2 * static_cast<double>(M) * static_cast<double>(N) *
          static_cast<double>(K);
  TFLOPS = FLOPS / 1e12;
}

void BenchmarkSgemm::SetHostValue() {
  for (int i = 0; i < A_size; ++i)
    A_host[i] = myblas::make_random<Element>(0, 1);
  for (int i = 0; i < B_size; ++i)
    B_host[i] = myblas::make_random<Element>(0, 1);

  memset(C_host.get(), 0, C_size * sizeof(Element));
}

void BenchmarkSgemm::SetUp(const ::benchmark::State& state) {
  M = N = K = static_cast<int>(state.range(0));

  GetTflops();

  A_size = M * K;
  B_size = K * N;
  C_size = M * N;
  A_host = std::make_unique<Element[]>(A_size);
  B_host = std::make_unique<Element[]>(B_size);
  C_host = std::make_unique<Element[]>(C_size);
  A_device = myblas::make_unique_cuda<Element, cudaMalloc>(A_size);
  B_device = myblas::make_unique_cuda<Element, cudaMalloc>(B_size);
  C_device = myblas::make_unique_cuda<Element, cudaMalloc>(C_size);

  SetHostValue();

  CHECK_CUBLAS(cublasCreate(&cublasHandle));
}

void BenchmarkSgemm::TearDown(const ::benchmark::State& state) {
  CHECK_CUBLAS(cublasDestroy(cublasHandle));
  A_device.reset(nullptr);
  B_device.reset(nullptr);
  C_device.reset(nullptr);
}

BENCHMARK_DEFINE_F(BenchmarkSgemm, cublas)(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    ResetDevice();
    CHECK_CUDA(cudaDeviceSynchronize());
    state.ResumeTiming();
    CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha, B_device.get(), N, A_device.get(), K,
                             &beta, C_device.get(), N));
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  state.counters["TFLOPS"] =
      benchmark::Counter(TFLOPS, benchmark::Counter::kIsIterationInvariantRate);
}

template <typename KernelId>
class BenchmarkSgemmKernel : public BenchmarkSgemm {
 public:
  BenchmarkSgemmKernel();

 protected:
  void BenchmarkCase(::benchmark::State&);

 private:
  static constexpr char const fixture_label_[] = "BenchmarkSgemmKernel";
};

template <typename KernelId>
BenchmarkSgemmKernel<KernelId>::BenchmarkSgemmKernel() {
  this->SetName(std::string(fixture_label_) + "/" +
                std::string(KernelId::label));
}

template <typename KernelId>
void BenchmarkSgemmKernel<KernelId>::BenchmarkCase(::benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    ResetDevice();
    CHECK_CUDA(cudaDeviceSynchronize());
    state.ResumeTiming();
    myblas::sgemm::sgemm<KernelId>(M, N, K, A_device.get(), B_device.get(),
                                   C_device.get());
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  state.counters["TFLOPS"] =
      benchmark::Counter(TFLOPS, benchmark::Counter::kIsIterationInvariantRate);
}

#define BENCHMARK_SGEMM_ARGS         \
  ->DenseRange(1024, 1024 * 2, 1024) \
      ->Iterations(1)                \
      ->UseRealTime()                \
      ->Unit(benchmark::kMillisecond);

}  // namespace

int main(int argc, char** argv) {
  char arg0_default[] = "benchmark";
  char* args_default = arg0_default;
  if (!argv) {
    argc = 1;
    argv = &args_default;
  }
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

  BENCHMARK_REGISTER_F(BenchmarkSgemm, cublas)
  BENCHMARK_SGEMM_ARGS

  ::benchmark::internal::RegisterBenchmarkInternal(
      new BenchmarkSgemmKernel<myblas::sgemm::KernelId1>) BENCHMARK_SGEMM_ARGS

      ::benchmark::internal::RegisterBenchmarkInternal(
          new BenchmarkSgemmKernel<myblas::sgemm::KernelId2>)
          BENCHMARK_SGEMM_ARGS

      ::benchmark::internal::RegisterBenchmarkInternal(
          new BenchmarkSgemmKernel<myblas::sgemm::KernelId3>)
          BENCHMARK_SGEMM_ARGS

      ::benchmark::internal::RegisterBenchmarkInternal(
          new BenchmarkSgemmKernel<myblas::sgemm::KernelId4>)
          BENCHMARK_SGEMM_ARGS

      ::benchmark::internal::RegisterBenchmarkInternal(
          new BenchmarkSgemmKernel<myblas::sgemm::KernelId5>)
          BENCHMARK_SGEMM_ARGS

      ::benchmark::internal::RegisterBenchmarkInternal(
          new BenchmarkSgemmKernel<myblas::sgemm::KernelId6>)
          BENCHMARK_SGEMM_ARGS

      ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}

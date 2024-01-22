#include "myblas_internal/helper_macros.h"

namespace myblas::sgemm {

__global__ void sgemmKernel1(const int M, const int N, const int K,
                             const float *__restrict__ A,
                             const float *__restrict__ B,
                             float *__restrict__ C) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  const int threadRow = threadIdx.x / (BN / TN);
  const int threadCol = threadIdx.x % (BN / TN);

  __shared__ float sharedA[BM * BK];
  __shared__ float sharedB[BK * BN];

  float rstEachThread[TM * TN] = {0.0};
  float vectorOuterProdA[TM] = {0.0};
  float vectorOuterProdB[TN] = {0.0};

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BM;

  const int SMEMLoadARow = threadIdx.x / BK;
  const int SMEMLoadACol = threadIdx.x % BK;
  const int SMEMLoadAStride = blockDim.x / BK;
  const int SMEMLoadBRow = threadIdx.x / BN;
  const int SMEMLoadBCol = threadIdx.x % BN;
  const int SMEMLoadBStride = blockDim.x / BN;

  for (int dotIdx = 0; dotIdx < (K - 1) / BK + 1; dotIdx++) {
    int dotOffset = dotIdx * BK;
#pragma unroll
    for (int loadOffset = 0; loadOffset < BM; loadOffset += SMEMLoadAStride) {
      if (blockIdx.y * BM + SMEMLoadARow + loadOffset < M &&
          dotOffset + SMEMLoadACol < K)
        sharedA[OFFSET(SMEMLoadARow + loadOffset, SMEMLoadACol, BK)] =
            A[OFFSET(SMEMLoadARow + loadOffset, SMEMLoadACol + dotOffset, K)];
      else
        sharedA[OFFSET(SMEMLoadARow + loadOffset, SMEMLoadACol, BK)] = 0.0;
    }
#pragma unroll
    for (int loadOffset = 0; loadOffset < BK; loadOffset += SMEMLoadBStride) {
      if (dotOffset + SMEMLoadBRow + loadOffset < K &&
          blockIdx.x * BN + SMEMLoadBCol < N)
        sharedB[OFFSET(SMEMLoadBRow + loadOffset, SMEMLoadBCol, BN)] =
            B[OFFSET(dotOffset + SMEMLoadBRow + loadOffset, SMEMLoadBCol, N)];
      else
        sharedB[OFFSET(SMEMLoadBRow + loadOffset, SMEMLoadBCol, BN)] = 0.0;
    }
    __syncthreads();

#pragma unroll
    for (int innerIndex = 0; innerIndex < BK; innerIndex++) {
#pragma unroll
      for (int i = 0; i < TM; i++)
        vectorOuterProdA[i] =
            sharedA[OFFSET(threadRow * TM + i, innerIndex, BK)];
#pragma unroll
      for (int i = 0; i < TN; i++)
        vectorOuterProdB[i] =
            sharedB[OFFSET(innerIndex, threadCol * TN + i, BN)];
#pragma unroll
      for (int rstEachThreadRow = 0; rstEachThreadRow < TM; rstEachThreadRow++)
        for (int rstEachthreadCol = 0; rstEachthreadCol < TN;
             rstEachthreadCol++)
          rstEachThread[OFFSET(rstEachThreadRow, rstEachthreadCol, TN)] +=
              vectorOuterProdA[rstEachThreadRow] *
              vectorOuterProdB[rstEachthreadCol];
    }
    __syncthreads();
  }
#pragma unroll
  for (int rstEachThreadRow = 0; rstEachThreadRow < TM; rstEachThreadRow++)
#pragma unroll
    for (int rstEachThreadCol = 0; rstEachThreadCol < TN; rstEachThreadCol++)
      if ((blockIdx.y * BM + threadRow * TM + rstEachThreadRow) < M &&
          (blockIdx.x * BN + threadCol * TN + rstEachThreadCol) < N)
        C[OFFSET(threadRow * TM + rstEachThreadRow,
                 threadCol * TN + rstEachThreadCol, N)] =
            rstEachThread[OFFSET(rstEachThreadRow, rstEachThreadCol, TN)];
}

void sgemmV1(const int M, const int N, const int K, const float *A,
             const float *B, float *C) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int TM = 8;
  constexpr int TN = 8;
  constexpr dim3 block_size(BM * BN / (TM * TN));
  const dim3 grid_size((N - 1) / BN + 1, (M - 1) / BM + 1);

  sgemmKernel1<<<grid_size, block_size>>>(M, N, K, A, B, C);
}

}  // namespace myblas::sgemm

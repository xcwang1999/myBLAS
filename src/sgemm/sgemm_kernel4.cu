#include "myblas_internal/helper_macros.h"

namespace myblas::sgemm {

__global__ void sgemmKernel4(const int M, const int N, const int K,
                             float *__restrict__ A, float *__restrict__ B,
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

  float ldgRegA[4] = {0.0};
  float ldgRegB[4] = {0.0};

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BM;

  const int SMEMLoadARow = threadIdx.x / (BK / 4);
  const int SMEMLoadACol = threadIdx.x % (BK / 4);
  const int SMEMLoadBRow = threadIdx.x / (BN / 4);
  const int SMEMLoadBCol = threadIdx.x % (BN / 4);

  for (int dotIdx = 0; dotIdx < (K - 1) / BK + 1; dotIdx++) {
    int dotOffset = dotIdx * BK;

    FETCH_FLOAT4(ldgRegA[0]) =
        FETCH_FLOAT4(A[OFFSET(SMEMLoadARow, SMEMLoadACol * 4 + dotOffset, K)]);
    sharedA[OFFSET(SMEMLoadACol * 4 + 0, SMEMLoadARow, BM)] = ldgRegA[0];
    sharedA[OFFSET(SMEMLoadACol * 4 + 1, SMEMLoadARow, BM)] = ldgRegA[1];
    sharedA[OFFSET(SMEMLoadACol * 4 + 2, SMEMLoadARow, BM)] = ldgRegA[2];
    sharedA[OFFSET(SMEMLoadACol * 4 + 3, SMEMLoadARow, BM)] = ldgRegA[3];

    FETCH_FLOAT4(ldgRegB[0]) =
        FETCH_FLOAT4(B[OFFSET(SMEMLoadBRow + dotOffset, SMEMLoadBCol * 4, N)]);
    FETCH_FLOAT4(sharedB[OFFSET(SMEMLoadBRow, SMEMLoadBCol * 4, BN)]) =
        FETCH_FLOAT4(ldgRegB[0]);

    __syncthreads();

#pragma unroll
    for (int innerIndex = 0; innerIndex < BK; innerIndex++) {
      FETCH_FLOAT4(vectorOuterProdA[0]) =
          FETCH_FLOAT4(sharedA[OFFSET(innerIndex, threadRow * TM + 0, BM)]);
      FETCH_FLOAT4(vectorOuterProdA[4]) =
          FETCH_FLOAT4(sharedA[OFFSET(innerIndex, threadRow * TM + 4, BM)]);

      FETCH_FLOAT4(vectorOuterProdB[0]) =
          FETCH_FLOAT4(sharedB[OFFSET(innerIndex, threadCol * TN / 2, BN)]);
      FETCH_FLOAT4(vectorOuterProdB[4]) = FETCH_FLOAT4(
          sharedB[OFFSET(innerIndex, threadCol * TN / 2 + BN / 2, BN)]);
#pragma unroll
      for (int rstEachThreadRow = 0; rstEachThreadRow < TM; rstEachThreadRow++)
#pragma unroll
        for (int rstEachThreadCol = 0; rstEachThreadCol < TN;
             rstEachThreadCol++)
          rstEachThread[OFFSET(rstEachThreadRow, rstEachThreadCol, TN)] +=
              vectorOuterProdA[rstEachThreadRow] *
              vectorOuterProdB[rstEachThreadCol];
    }
    __syncthreads();
  }

#pragma unroll
  for (int resEachThreadRow = 0; resEachThreadRow < TM; resEachThreadRow++)
    FETCH_FLOAT4(
        C[OFFSET(threadRow * TM + resEachThreadRow, threadCol * TN / 2, N)]) =
        FETCH_FLOAT4(rstEachThread[OFFSET(resEachThreadRow, 0, TN)]);

#pragma unroll
  for (int resEachThreadRow = 0; resEachThreadRow < TM; resEachThreadRow++)
    FETCH_FLOAT4(C[OFFSET(threadRow * TM + resEachThreadRow,
                          threadCol * TN / 2 + BN / 2, N)]) =
        FETCH_FLOAT4(rstEachThread[OFFSET(resEachThreadRow, 4, TN)]);
}

void sgemmV4(const int M, const int N, const int K, float *A, float *B,
             float *C) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int TM = 8;
  constexpr int TN = 8;
  constexpr dim3 block_size(BM * BN / (TM * TN));
  const dim3 grid_size((N - 1) / BN + 1, (M - 1) / BM + 1);

  sgemmKernel4<<<grid_size, block_size>>>(M, N, K, A, B, C);
}

}  // namespace myblas::sgemm
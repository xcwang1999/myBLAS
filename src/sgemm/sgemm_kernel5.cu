#include "utils.h"

__global__ void sgemmKernel5(const int M, const int N, const int K,
                             float *__restrict__ A, float *__restrict__ B,
                             float *__restrict__ C) {
  const int BM = 128;
  const int BN = 128;
  const int BK = 8;
  const int TM = 8;
  const int TN = 8;
  const int threadRow = threadIdx.x / (BN / TN);
  const int threadCol = threadIdx.x % (BN / TN);

  __shared__ float sharedA[2][BM * BK];
  __shared__ float sharedB[2][BK * BN];
  float vectorOuterProdA[2][TM] = {0.0};
  float vectorOuterProdB[2][TN] = {0.0};
  float rstEachThread[TM * TN] = {0.0};

  float ldgRegA[4] = {0.0};
  float ldgRegB[4] = {0.0};

  const int SMEMLoadARow = threadIdx.x / (BK / 4);
  const int SMEMLoadACol = threadIdx.x % (BK / 4);
  const int SMEMLoadBRow = threadIdx.x / (BN / 4);
  const int SMEMLoadBCol = threadIdx.x % (BN / 4);

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  FETCH_FLOAT4(ldgRegA[0]) =
      FETCH_FLOAT4(A[OFFSET(SMEMLoadARow, SMEMLoadACol * 4, K)]);
  sharedA[0][OFFSET(SMEMLoadACol * 4 + 0, SMEMLoadARow, BM)] = ldgRegA[0];
  sharedA[0][OFFSET(SMEMLoadACol * 4 + 1, SMEMLoadARow, BM)] = ldgRegA[1];
  sharedA[0][OFFSET(SMEMLoadACol * 4 + 2, SMEMLoadARow, BM)] = ldgRegA[2];
  sharedA[0][OFFSET(SMEMLoadACol * 4 + 3, SMEMLoadARow, BM)] = ldgRegA[3];

  FETCH_FLOAT4(ldgRegB[0]) =
      FETCH_FLOAT4(B[OFFSET(SMEMLoadBRow, SMEMLoadBCol * 4, N)]);
  FETCH_FLOAT4(sharedB[0][OFFSET(SMEMLoadBRow, SMEMLoadBCol * 4, BN)]) =
      FETCH_FLOAT4(ldgRegB[0]);

  __syncthreads();

  int innerIndex = 0;
  FETCH_FLOAT4(vectorOuterProdA[0][0]) =
      FETCH_FLOAT4(sharedA[0][OFFSET(innerIndex, threadRow * TM + 0, BM)]);
  FETCH_FLOAT4(vectorOuterProdA[0][4]) =
      FETCH_FLOAT4(sharedA[0][OFFSET(innerIndex, threadRow * TM + 4, BM)]);

  FETCH_FLOAT4(vectorOuterProdB[0][0]) =
      FETCH_FLOAT4(sharedB[0][OFFSET(innerIndex, threadCol * TN / 2, BN)]);
  FETCH_FLOAT4(vectorOuterProdB[0][4]) = FETCH_FLOAT4(
      sharedB[0][OFFSET(innerIndex, threadCol * TN / 2 + BN / 2, BN)]);

  int writeIndex = 1;
  int loadIndex;

  for (int dotIndex = 1; dotIndex <= (K - 1) / BK + 1; dotIndex++) {
    int dotOffset = BK * dotIndex;
    if (dotIndex < (K - 1) / BK + 1) {
      FETCH_FLOAT4(ldgRegA[0]) = FETCH_FLOAT4(
          A[OFFSET(SMEMLoadARow, SMEMLoadACol * 4 + dotOffset, K)]);

      FETCH_FLOAT4(ldgRegB[0]) = FETCH_FLOAT4(
          B[OFFSET(dotOffset + SMEMLoadBRow, SMEMLoadBCol * 4, N)]);
    }

    loadIndex = writeIndex ^ 1;

#pragma unroll
    for (int innerIndex = 1; innerIndex < BK; innerIndex++) {
      FETCH_FLOAT4(vectorOuterProdA[innerIndex % 2][0]) = FETCH_FLOAT4(
          sharedA[loadIndex][OFFSET(innerIndex, threadRow * TM + 0, BM)]);
      FETCH_FLOAT4(vectorOuterProdA[innerIndex % 2][4]) = FETCH_FLOAT4(
          sharedA[loadIndex][OFFSET(innerIndex, threadRow * TM + 4, BM)]);

      FETCH_FLOAT4(vectorOuterProdB[innerIndex % 2][0]) = FETCH_FLOAT4(
          sharedB[loadIndex][OFFSET(innerIndex, threadCol * TN / 2, BN)]);
      FETCH_FLOAT4(vectorOuterProdB[innerIndex % 2][4]) =
          FETCH_FLOAT4(sharedB[loadIndex][OFFSET(
              innerIndex, threadCol * TN / 2 + BN / 2, BN)]);

#pragma unroll
      for (int resEachThreadRow = 0; resEachThreadRow < TM; resEachThreadRow++)
#pragma unroll
        for (int resEachThreadCol = 0; resEachThreadCol < TN;
             resEachThreadCol++)
          rstEachThread[OFFSET(resEachThreadRow, resEachThreadCol, TN)] +=
              vectorOuterProdA[(innerIndex - 1) % 2][resEachThreadRow] *
              vectorOuterProdB[(innerIndex - 1) % 2][resEachThreadCol];
    }

    if (dotIndex < (K - 1) / BK + 1) {
      sharedA[writeIndex][OFFSET(SMEMLoadACol * 4 + 0, SMEMLoadARow, BM)] =
          ldgRegA[0];
      sharedA[writeIndex][OFFSET(SMEMLoadACol * 4 + 1, SMEMLoadARow, BM)] =
          ldgRegA[1];
      sharedA[writeIndex][OFFSET(SMEMLoadACol * 4 + 2, SMEMLoadARow, BM)] =
          ldgRegA[2];
      sharedA[writeIndex][OFFSET(SMEMLoadACol * 4 + 3, SMEMLoadARow, BM)] =
          ldgRegA[3];

      FETCH_FLOAT4(
          sharedB[writeIndex][OFFSET(SMEMLoadBRow, SMEMLoadBCol * 4, BN)]) =
          FETCH_FLOAT4(ldgRegB[0]);
    }

    __syncthreads();

    FETCH_FLOAT4(vectorOuterProdA[0][0]) = FETCH_FLOAT4(
        sharedA[writeIndex][OFFSET(innerIndex, threadRow * TM + 0, BM)]);
    FETCH_FLOAT4(vectorOuterProdA[0][4]) = FETCH_FLOAT4(
        sharedA[writeIndex][OFFSET(innerIndex, threadRow * TM + 4, BM)]);

    FETCH_FLOAT4(vectorOuterProdB[0][0]) = FETCH_FLOAT4(
        sharedB[writeIndex][OFFSET(innerIndex, threadCol * TN / 2, BN)]);
    FETCH_FLOAT4(vectorOuterProdB[0][4]) =
        FETCH_FLOAT4(sharedB[writeIndex][OFFSET(
            innerIndex, threadCol * TN / 2 + BN / 2, BN)]);

#pragma unroll
    for (int resEachThreadRow = 0; resEachThreadRow < TM; resEachThreadRow++)
#pragma unroll
      for (int resEachThreadCol = 0; resEachThreadCol < TN; resEachThreadCol++)
        rstEachThread[OFFSET(resEachThreadRow, resEachThreadCol, TN)] +=
            vectorOuterProdA[(BK - 1) % 2][resEachThreadRow] *
            vectorOuterProdB[(BK - 1) % 2][resEachThreadCol];
    writeIndex ^= 1;
  }

  for (int resEachThreadRow = 0; resEachThreadRow < TM; resEachThreadRow++)
    FETCH_FLOAT4(
        C[OFFSET(threadRow * TM + resEachThreadRow, threadCol * TN / 2, N)]) =
        FETCH_FLOAT4(rstEachThread[OFFSET(resEachThreadRow, 0, TN)]);

  for (int resEachThreadRow = 0; resEachThreadRow < TM; resEachThreadRow++)
    FETCH_FLOAT4(C[OFFSET(threadRow * TM + resEachThreadRow,
                          threadCol * TN / 2 + BN / 2, N)]) =
        FETCH_FLOAT4(rstEachThread[OFFSET(resEachThreadRow, 4, TN)]);
}

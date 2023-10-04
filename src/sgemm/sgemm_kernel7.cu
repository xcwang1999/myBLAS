#include "utils.h"

__global__ void sgemmKernel7(const int M, const int K, float *__restrict__ A,
                             const size_t pitchA, float *__restrict__ B,
                             const size_t pitchB, float *__restrict__ C,
                             const size_t pitchC) {
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

  const int ldA = pitchA / sizeof(float);
  const int ldB = pitchB / sizeof(float);
  const int ldC = pitchC / sizeof(float);

  A += blockIdx.y * BM * ldA;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * ldB + blockIdx.x * BN;

  if (blockIdx.y * BM + SMEMLoadARow < M)
    FETCH_FLOAT4(ldgRegA[0]) =
        FETCH_FLOAT4(A[OFFSET(SMEMLoadARow, SMEMLoadACol * 4, ldA)]);

  sharedA[0][OFFSET(SMEMLoadACol * 4 + 0, SMEMLoadARow, BM)] = ldgRegA[0];
  sharedA[0][OFFSET(SMEMLoadACol * 4 + 1, SMEMLoadARow, BM)] = ldgRegA[1];
  sharedA[0][OFFSET(SMEMLoadACol * 4 + 2, SMEMLoadARow, BM)] = ldgRegA[2];
  sharedA[0][OFFSET(SMEMLoadACol * 4 + 3, SMEMLoadARow, BM)] = ldgRegA[3];

  int dotOffset = 0 * BK;

  if (dotOffset + SMEMLoadBRow < K)
    FETCH_FLOAT4(ldgRegB[0]) =
        FETCH_FLOAT4(B[OFFSET(SMEMLoadBRow, SMEMLoadBCol * 4, ldB)]);

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

  for (int dotIndex = 1; dotIndex <= ldA / BK; dotIndex++) {
    int dotOffset = BK * dotIndex;
    float ldgRegA[4] = {0.0};
    float ldgRegB[4] = {0.0};
    if (dotIndex < ldA / BK) {
      if (blockIdx.y * BM + SMEMLoadARow < M)
        FETCH_FLOAT4(ldgRegA[0]) = FETCH_FLOAT4(
            A[OFFSET(SMEMLoadARow, SMEMLoadACol * 4 + dotOffset, ldA)]);

      if (dotOffset + SMEMLoadBRow < K)
        FETCH_FLOAT4(ldgRegB[0]) = FETCH_FLOAT4(
            B[OFFSET(dotOffset + SMEMLoadBRow, SMEMLoadBCol * 4, ldB)]);
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
      for (int rstEachThreadRow = 0; rstEachThreadRow < TM; rstEachThreadRow++)
#pragma unroll
        for (int rstEachThreadCol = 0; rstEachThreadCol < TN;
             rstEachThreadCol++)
          rstEachThread[OFFSET(rstEachThreadRow, rstEachThreadCol, TN)] +=
              vectorOuterProdA[(innerIndex - 1) % 2][rstEachThreadRow] *
              vectorOuterProdB[(innerIndex - 1) % 2][rstEachThreadCol];
    }

    if (dotIndex < ldA / BK) {
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
    for (int rstEachThreadRow = 0; rstEachThreadRow < TM; rstEachThreadRow++)
#pragma unroll
      for (int rstEachThreadCol = 0; rstEachThreadCol < TN; rstEachThreadCol++)
        rstEachThread[OFFSET(rstEachThreadRow, rstEachThreadCol, TN)] +=
            vectorOuterProdA[(BK - 1) % 2][rstEachThreadRow] *
            vectorOuterProdB[(BK - 1) % 2][rstEachThreadCol];
    writeIndex ^= 1;
  }

#pragma unroll
  for (int rstEachThreadRow = 0; rstEachThreadRow < TM; rstEachThreadRow++) {
    if (blockIdx.y * BM + threadRow * TM + rstEachThreadRow < M)
      FETCH_FLOAT4(C[OFFSET(threadRow * TM + rstEachThreadRow,
                            threadCol * TN / 2, ldC)]) =
          FETCH_FLOAT4(rstEachThread[OFFSET(rstEachThreadRow, 0, TN)]);
  }
#pragma unroll
  for (int rstEachThreadRow = 0; rstEachThreadRow < TM; rstEachThreadRow++) {
    if (blockIdx.y * BM + threadRow * TM + rstEachThreadRow < M)
      FETCH_FLOAT4(C[OFFSET(threadRow * TM + rstEachThreadRow,
                            threadCol * TN / 2 + BN / 2, ldC)]) =
          FETCH_FLOAT4(rstEachThread[OFFSET(rstEachThreadRow, 4, TN)]);
  }
}

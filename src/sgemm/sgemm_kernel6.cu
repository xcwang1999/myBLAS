#include "utils.h"

__global__ void sgemmKernel6(const int M, const int N, const int K,
                             const float *__restrict__ A,
                             const float *__restrict__ B,
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

  const int SMEMLoadARow = threadIdx.x / (BK / 4);
  const int SMEMLoadACol = threadIdx.x % (BK / 4);
  const int SMEMLoadBRow = threadIdx.x / (BN / 4);
  const int SMEMLoadBCol = threadIdx.x % (BN / 4);

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  int dotOffset = 0 * BK;
  float regA[4] = {0.0};
  float regB[4] = {0.0};

  if (blockIdx.y * BM + SMEMLoadARow < M &&
      dotOffset + SMEMLoadACol * 4 + 3 < K) {
    regA[0] = A[OFFSET(SMEMLoadARow, SMEMLoadACol * 4 + dotOffset + 0, K)];
    regA[1] = A[OFFSET(SMEMLoadARow, SMEMLoadACol * 4 + dotOffset + 1, K)];
    regA[2] = A[OFFSET(SMEMLoadARow, SMEMLoadACol * 4 + dotOffset + 2, K)];
    regA[3] = A[OFFSET(SMEMLoadARow, SMEMLoadACol * 4 + dotOffset + 3, K)];
  }
  if ((blockIdx.y * BM + SMEMLoadARow < M) &&
      (K <= dotOffset + SMEMLoadACol * 4 + 3) &&
      (dotOffset + SMEMLoadACol * 4 < K)) {
    for (int i = 0; i < K % 4; i++)
      regA[i] = A[OFFSET(SMEMLoadARow, SMEMLoadACol * 4 + dotOffset + i, K)];
  }
  sharedA[0][OFFSET(SMEMLoadACol * 4 + 0, SMEMLoadARow, BM)] = regA[0];
  sharedA[0][OFFSET(SMEMLoadACol * 4 + 1, SMEMLoadARow, BM)] = regA[1];
  sharedA[0][OFFSET(SMEMLoadACol * 4 + 2, SMEMLoadARow, BM)] = regA[2];
  sharedA[0][OFFSET(SMEMLoadACol * 4 + 3, SMEMLoadARow, BM)] = regA[3];

  if (dotOffset + SMEMLoadBRow < K &&
      blockIdx.x * BN + SMEMLoadBCol * 4 + 3 < N) {
    regB[0] = B[OFFSET(SMEMLoadBRow + dotOffset, SMEMLoadBCol * 4 + 0, N)];
    regB[1] = B[OFFSET(SMEMLoadBRow + dotOffset, SMEMLoadBCol * 4 + 1, N)];
    regB[2] = B[OFFSET(SMEMLoadBRow + dotOffset, SMEMLoadBCol * 4 + 2, N)];
    regB[3] = B[OFFSET(SMEMLoadBRow + dotOffset, SMEMLoadBCol * 4 + 3, N)];
  }
  if ((dotOffset + SMEMLoadBRow < K) &&
      (N <= blockIdx.x * BN + SMEMLoadBCol * 4 + 3) &&
      (blockIdx.x * BN + SMEMLoadBCol * 4 < N)) {
    for (int i = 0; i < N % 4; i++)
      regB[i] = B[OFFSET(SMEMLoadBRow + dotOffset, SMEMLoadBCol * 4 + i, N)];
    ;
  }
  FETCH_FLOAT4(sharedB[0][OFFSET(SMEMLoadBRow, SMEMLoadBCol * 4, BN)]) =
      FETCH_FLOAT4(regB[0]);

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
    int dotOffset = dotIndex * BK;
    float regA[4] = {0.0};
    float regB[4] = {0.0};
    if (dotIndex < (K - 1) / BK + 1) {
      if (blockIdx.y * BM + SMEMLoadARow < M &&
          dotOffset + SMEMLoadACol * 4 + 3 < K) {
        regA[0] = A[OFFSET(SMEMLoadARow, SMEMLoadACol * 4 + dotOffset + 0, K)];
        regA[1] = A[OFFSET(SMEMLoadARow, SMEMLoadACol * 4 + dotOffset + 1, K)];
        regA[2] = A[OFFSET(SMEMLoadARow, SMEMLoadACol * 4 + dotOffset + 2, K)];
        regA[3] = A[OFFSET(SMEMLoadARow, SMEMLoadACol * 4 + dotOffset + 3, K)];
      }
      if ((blockIdx.y * BM + SMEMLoadARow < M) &&
          (K <= dotOffset + SMEMLoadACol * 4 + 3) &&
          (dotOffset + SMEMLoadACol * 4 < K)) {
        for (int i = 0; i < K % 4; i++)
          regA[i] =
              A[OFFSET(SMEMLoadARow, SMEMLoadACol * 4 + dotOffset + i, K)];
      }

      if (dotOffset + SMEMLoadBRow < K &&
          blockIdx.x * BN + SMEMLoadBCol * 4 + 3 < N) {
        regB[0] = B[OFFSET(SMEMLoadBRow + dotOffset, SMEMLoadBCol * 4 + 0, N)];
        regB[1] = B[OFFSET(SMEMLoadBRow + dotOffset, SMEMLoadBCol * 4 + 1, N)];
        regB[2] = B[OFFSET(SMEMLoadBRow + dotOffset, SMEMLoadBCol * 4 + 2, N)];
        regB[3] = B[OFFSET(SMEMLoadBRow + dotOffset, SMEMLoadBCol * 4 + 3, N)];
      }
      if ((dotOffset + SMEMLoadBRow < K) &&
          (N <= blockIdx.x * BN + SMEMLoadBCol * 4 + 3) &&
          (blockIdx.x * BN + SMEMLoadBCol * 4 < N)) {
        for (int i = 0; i < N % 4; i++)
          regB[i] =
              B[OFFSET(SMEMLoadBRow + dotOffset, SMEMLoadBCol * 4 + i, N)];
        ;
      }
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
          regA[0];
      sharedA[writeIndex][OFFSET(SMEMLoadACol * 4 + 1, SMEMLoadARow, BM)] =
          regA[1];
      sharedA[writeIndex][OFFSET(SMEMLoadACol * 4 + 2, SMEMLoadARow, BM)] =
          regA[2];
      sharedA[writeIndex][OFFSET(SMEMLoadACol * 4 + 3, SMEMLoadARow, BM)] =
          regA[3];

      FETCH_FLOAT4(
          sharedB[writeIndex][OFFSET(SMEMLoadBRow, SMEMLoadBCol * 4, BN)]) =
          FETCH_FLOAT4(regB[0]);
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

#pragma unroll
  for (int resEachThreadRow = 0; resEachThreadRow < TM; resEachThreadRow++) {
    if ((blockIdx.y * BM + threadRow * TM + resEachThreadRow < M) &&
        blockIdx.x * BN + threadCol * TN / 2 + 3 < N) {
      C[OFFSET(threadRow * TM + resEachThreadRow, threadCol * TN / 2 + 0, N)] =
          rstEachThread[OFFSET(resEachThreadRow, 0, TN)];
      C[OFFSET(threadRow * TM + resEachThreadRow, threadCol * TN / 2 + 1, N)] =
          rstEachThread[OFFSET(resEachThreadRow, 1, TN)];
      C[OFFSET(threadRow * TM + resEachThreadRow, threadCol * TN / 2 + 2, N)] =
          rstEachThread[OFFSET(resEachThreadRow, 2, TN)];
      C[OFFSET(threadRow * TM + resEachThreadRow, threadCol * TN / 2 + 3, N)] =
          rstEachThread[OFFSET(resEachThreadRow, 3, TN)];
    }
    if ((blockIdx.y * BM + threadRow * TM + resEachThreadRow < M) &&
        (N <= blockIdx.x * BN + threadCol * TN / 2 + 3) &&
        (blockIdx.x * BN + threadCol * TN / 2 < N)) {
      for (int i = 0; i < N % 4; i++)
        C[OFFSET(threadRow * TM + resEachThreadRow, threadCol * TN / 2 + i,
                 N)] = rstEachThread[OFFSET(resEachThreadRow, i, TN)];
    }
  }

#pragma unroll
  for (int resEachThreadRow = 0; resEachThreadRow < TM; resEachThreadRow++) {
    if (blockIdx.y * BM + threadRow * TM + resEachThreadRow < M &&
        blockIdx.x * BN + threadCol * TN / 2 + BN / 2 + 3 < N) {
      C[OFFSET(threadRow * TM + resEachThreadRow,
               threadCol * TN / 2 + BN / 2 + 0, N)] =
          rstEachThread[OFFSET(resEachThreadRow, TN / 2 + 0, TN)];
      C[OFFSET(threadRow * TM + resEachThreadRow,
               threadCol * TN / 2 + BN / 2 + 1, N)] =
          rstEachThread[OFFSET(resEachThreadRow, TN / 2 + 1, TN)];
      C[OFFSET(threadRow * TM + resEachThreadRow,
               threadCol * TN / 2 + BN / 2 + 2, N)] =
          rstEachThread[OFFSET(resEachThreadRow, TN / 2 + 2, TN)];
      C[OFFSET(threadRow * TM + resEachThreadRow,
               threadCol * TN / 2 + BN / 2 + 3, N)] =
          rstEachThread[OFFSET(resEachThreadRow, TN / 2 + 3, TN)];
    }
    if ((blockIdx.y * BM + threadRow * TM + resEachThreadRow < M) &&
        (N <= blockIdx.x * BN + threadCol * TN / 2 + BN / 2 + 3) &&
        (blockIdx.x * BN + threadCol * TN / 2 + BN / 2 < N)) {
      for (int i = 0; i < N % 4; i++)
        C[OFFSET(threadRow * TM + resEachThreadRow,
                 threadCol * TN / 2 + BN / 2 + i, N)] =
            rstEachThread[OFFSET(resEachThreadRow, TN / 2 + i, TN)];
    }
  }
}

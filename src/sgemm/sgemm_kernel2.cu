__global__ void sgemmKernel2(const int M, const int N, const int K,
                             float *__restrict__ A, float *__restrict__ B,
                             float *__restrict__ C) {
  const int BM = 128;
  const int BN = 128;
  const int BK = 8;
  const int TM = 8;
  const int TN = 8;
  const int threadRow = threadIdx.x / (BN / TN);
  const int threadCol = threadIdx.x % (BN / TN);

  __shared__ float sharedA[BM * BK];
  __shared__ float sharedB[BK * BN];

  float resultPerThread[TM * TN] = {0.0};
  float vectorOuterA[TM] = {0.0};
  float vectorOuterB[TN] = {0.0};

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BM;

  const int sharedARow = threadIdx.x / (BK / 4);
  const int sharedACol = threadIdx.x % (BK / 4);
  const int sharedBRow = threadIdx.x / (BN / 4);
  const int sharedBCol = threadIdx.x % (BN / 4);

  for (int dotOrder = 0; dotOrder < K; dotOrder += BK) {
    reinterpret_cast<float4 *>(&sharedA[sharedARow * BK + sharedACol * 4])[0] =
        reinterpret_cast<float4 *>(&A[sharedARow * K + sharedACol * 4])[0];

    reinterpret_cast<float4 *>(&sharedB[sharedBRow * BN + sharedBCol * 4])[0] =
        reinterpret_cast<float4 *>(&B[sharedBRow * N + sharedBCol * 4])[0];
    __syncthreads();

    A += BK;
    B += BK * N;

#pragma unroll
    for (int innerOuterProdOrder = 0; innerOuterProdOrder < BK;
         innerOuterProdOrder++) {
#pragma unroll
      for (int i = 0; i < TM; i++)
        vectorOuterA[i] =
            sharedA[(threadRow * TM + i) * BK + innerOuterProdOrder];
#pragma unroll
      for (int i = 0; i < TN; i++)
        vectorOuterB[i] =
            sharedB[innerOuterProdOrder * BN + (threadCol * TN + i)];
#pragma unroll
      for (int resultRow = 0; resultRow < TM; resultRow++)
        for (int resultCol = 0; resultCol < TN; resultCol++)
          resultPerThread[resultRow * TN + resultCol] +=
              vectorOuterA[resultRow] * vectorOuterB[resultCol];
    }
    __syncthreads();
  }

#pragma unroll
  for (int resultRow = 0; resultRow < TM; resultRow += 1)
#pragma unroll
    for (int resultCol = 0; resultCol < TN; resultCol += 4) {
      float4 tmp;
      tmp.x = resultPerThread[resultRow * TN + resultCol + 0];
      tmp.y = resultPerThread[resultRow * TN + resultCol + 1];
      tmp.z = resultPerThread[resultRow * TN + resultCol + 2];
      tmp.w = resultPerThread[resultRow * TN + resultCol + 3];
      reinterpret_cast<float4 *>(&C[(threadRow * TM + resultRow) * N +
                                    (threadCol * TN + resultCol)])[0] = tmp;
    }
}
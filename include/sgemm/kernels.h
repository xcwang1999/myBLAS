#pragma once

namespace myblas::sgemm {

void sgemmV1(const int M, const int N, const int K, const float* A,
             const float* B, float* C);
void sgemmV2(const int M, const int N, const int K, float* A, float* B,
             float* C);
void sgemmV3(const int M, const int N, const int K, float* A, float* B,
             float* C);
void sgemmV4(const int M, const int N, const int K, float* A, float* B,
             float* C);
void sgemmV5(const int M, const int N, const int K, float* A, float* B,
             float* C);
void sgemmV6(const int M, const int N, const int K, const float* A,
             const float* B, float* C);

}  // namespace myblas::sgemm
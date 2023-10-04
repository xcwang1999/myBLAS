#include <mma.h>

#include "utils.h"

namespace wmma = nvcuda::wmma;

__global__ void hgemmKernel1(const int M, const int N, const int K,
                             half __restrict__ *A, const size_t pitchA,
                             half __restrict__ *B, const size_t pitchB,
                             half __restrict__ *C, const size_t pitchC) {
  const int BM = 128;
  const int BN = 256;
  const int BK = 32;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = threadIdx.x;
  int wid = tid >> 5;

  const int APAD = 8;
  const int BPAD = 8;

  __shared__ half s_a[BM][BK + APAD];
  __shared__ half s_b[BK][BN + BPAD];

  const int ldA = pitchA / sizeof(half);
  const int ldB = pitchB / sizeof(half);
  const int ldC = pitchC / sizeof(half);
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2]
                                                                          [4];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2]
                                                                          [4];
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

#pragma unroll
  for (int i = 0; i < 4; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      wmma::fill_fragment(frag_c[i][j], 0.0);
    }
  }

  int load_a_smem_m = (tid >> 2) << 1;
  int load_a_smem_k = (tid & 3) << 3;
  int load_b_smem_k = (tid >> 5) << 2;
  int load_b_smem_n = (tid & 31) << 3;

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, ldA);
  int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, ldB);

  int comp_c_frag_m = wid & 1;
  int comp_c_frag_n = wid >> 1;

  for (int bk = 0; bk < ldA / BK; bk++) {
    FETCH_FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) =
        FETCH_FLOAT4(A[load_a_gmem_addr]);
    FETCH_FLOAT4(s_a[load_a_smem_m + 1][load_a_smem_k]) =
        FETCH_FLOAT4(A[load_a_gmem_addr + ldA]);
    FETCH_FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) =
        FETCH_FLOAT4(B[load_b_gmem_addr]);
    FETCH_FLOAT4(s_b[load_b_smem_k + 1][load_b_smem_n]) =
        FETCH_FLOAT4(B[load_b_gmem_addr + ldB]);
    FETCH_FLOAT4(s_b[load_b_smem_k + 2][load_b_smem_n]) =
        FETCH_FLOAT4(B[load_b_gmem_addr + 2 * ldB]);
    FETCH_FLOAT4(s_b[load_b_smem_k + 3][load_b_smem_n]) =
        FETCH_FLOAT4(B[load_b_gmem_addr + 3 * ldB]);

    load_a_gmem_addr += BK;
    load_b_gmem_addr += BK * ldB;

    __syncthreads();

    wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64][0],
                           BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][0],
                           BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][0],
                           BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][0],
                           BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64][16],
                           BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16],
                           BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16],
                           BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16],
                           BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &s_b[0][comp_c_frag_n * 64],
                           BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &s_b[0][comp_c_frag_n * 64 + 16],
                           BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &s_b[0][comp_c_frag_n * 64 + 32],
                           BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &s_b[0][comp_c_frag_n * 64 + 48],
                           BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64],
                           BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16],
                           BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32],
                           BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48],
                           BN + BPAD);

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
      }
    }

    __syncthreads();
  }

  int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
  int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
  int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, ldC);
#pragma unroll
  for (int i = 0; i < 4; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      wmma::store_matrix_sync(&C[store_c_gmem_addr + i * 16 * ldC + j * 16],
                              frag_c[i][j], ldC, wmma::mem_row_major);
    }
  }
}
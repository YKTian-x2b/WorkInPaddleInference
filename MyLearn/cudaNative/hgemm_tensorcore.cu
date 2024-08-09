#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cublas_v2.h"
#include <mma.h>

#include <random>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <cstdio>

#include "common.cuh"

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(x) (reinterpret_cast<float4*>(&(x))[0])

__host__ void assignData(const int M, const int N, const int K, half* A, half* B, half* C, half* ref_C){
    std::default_random_engine e;
    e.seed(time(0));
    std::uniform_real_distribution<float> u(1.5, 4.5);
    for(size_t i = 0; i < M; i++){
        for(size_t j = 0; j < K; j++){          
          // A[i * K + j] = __float2half(1.);
          A[i * K + j] = __float2half(u(e));
        }
    }
    for(size_t i = 0; i < K; i++){
        for(size_t j = 0; j < N; j++){
          // B[i * N + j] = __float2half(1.);
          B[i * N + j] = __float2half(u(e));
        }
    }
    memset(C, 0, sizeof(half) * M * N);
    memset(ref_C, 0, sizeof(half) * M * N);
}

template<int M, int N, int K>
void cpuGemm(half *a, half *b, half *c) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += (float)a[OFFSET(m, k, K)] * (float)b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = (half)psum;
        }
    }
}

void valid(const int M, const int N, half* my_c, half* ref_c){
  float max_diff = 0.f;
  bool succ = true; 
  for(int m = 0; m < M; m++){
    for(int n = 0; n < N; n++){
      float my_c_ele = __half2float(my_c[m*N+n]);
      float ref_c_ele = __half2float(ref_c[m*N+n]);
      float diff = std::abs(my_c_ele - ref_c_ele);  
      if (diff / ref_c_ele > 0.01) {  
        succ = false;
        printf("Error: [%d, %d],     my_c_ele: %f,     ref_c_ele: %f\n", m, n, my_c_ele, ref_c_ele);
      } 
      max_diff = std::max(max_diff, diff);      
    }
  }
  printf("max_diff:%f \n", max_diff);
  if(succ)  printf("Success \n");
}

void print(const int M, const int N, half* my_c){
  for(int m = 0; m < M; m++){
    for(int n = 0; n < N; n++){
      float my_c_ele = __half2float(my_c[m*N+n]); 
      printf("[%d, %d], my_c_ele: %f;    ", m, n, my_c_ele);
    }
    printf( "\n" );
  }
  printf( "\n" );
}


template<int M, int N, int K, int BM, int BN, int BK>
__global__ void HGEMMV1(half *a, half *b, half *c){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = threadIdx.x;
  int wid = tid >> 5;

  const int APAD = 8;
  const int BPAD = 8;
  __shared__ half smem_a[BM][BK + APAD];
  __shared__ half smem_b[BK][BN + BPAD];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
          wmma::fill_fragment(frag_c[i][j], 0.0);
      }
  }

  // 连续4个线程处理同一行; <<1表示一行4个线程会处理连续两行
  int load_a_smem_m = (tid >> 2) << 1;
  // 连续4个线程按 0 8 16 24 排列  (tid%4) * 8
  int load_a_smem_k = (tid & 3) << 3;
  // 连续32个线程处理同一行; <<2表示一行32个线程会处理连续四行
  int load_b_smem_k = (tid >> 5) << 2;
  // 连续32个线程按 0 8 16 24 32... 排列  (tid%32) * 8
  int load_b_smem_n = (tid & 31) << 3;

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
  int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

  int comp_c_frag_m = wid & 1;
  int comp_c_frag_n = wid >> 1;

  for(int bk = 0; bk < K / BK; bk++){
    FLOAT4(smem_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(smem_a[load_a_smem_m + 1][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr + K]);
    FLOAT4(smem_b[load_b_smem_k    ][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr        ]);
    FLOAT4(smem_b[load_b_smem_k + 1][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr +     N]);
    FLOAT4(smem_b[load_b_smem_k + 2][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + 2 * N]);
    FLOAT4(smem_b[load_b_smem_k + 3][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + 3 * N]);

    load_a_gmem_addr += BK;
    load_b_gmem_addr += BK * N;

    __syncthreads();

    wmma::load_matrix_sync(frag_a[0][0], &smem_a[comp_c_frag_m * 64     ][ 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &smem_a[comp_c_frag_m * 64 + 16][ 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &smem_a[comp_c_frag_m * 64 + 32][ 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &smem_a[comp_c_frag_m * 64 + 48][ 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &smem_a[comp_c_frag_m * 64     ][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &smem_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &smem_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &smem_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &smem_b[ 0][comp_c_frag_n * 64     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &smem_b[ 0][comp_c_frag_n * 64 + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &smem_b[ 0][comp_c_frag_n * 64 + 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &smem_b[ 0][comp_c_frag_n * 64 + 48], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &smem_b[16][comp_c_frag_n * 64     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &smem_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &smem_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &smem_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
      }
    }

    __syncthreads();
  }

  int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
  int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
  int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

  for(int i = 0; i < 4; i++){
    for(int j = 0; j < 4; j++){
      wmma::store_matrix_sync(&c[store_c_gmem_addr + OFFSET(i,j,N) * 16], frag_c[i][j], N, wmma::mem_row_major);
    }
  }

}


template<int M, int N, int K, int BM, int BN, int BK>
__global__ void HGEMMV2(half *a, half *b, half *c){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = threadIdx.x;
  int wid = tid >> 5;

  const int APAD = 8;
  const int BPAD = 8;
  __shared__ half smem_a[BM][BK + APAD];
  __shared__ half smem_b[BK][BN + BPAD];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
          wmma::fill_fragment(frag_c[i][j], 0.0);
      }
  }
  // 连续4个线程处理同一行; <<1表示一行4个线程会处理连续两行
  int load_a_smem_m = (tid >> 2) << 1;
  // 连续4个线程按 0 8 16 24 排列  (tid%4) * 8
  int load_a_smem_k = (tid & 3) << 3;
  // 连续32个线程处理同一行; <<2表示一行32个线程会处理连续四行
  int load_b_smem_k = (tid >> 5) << 2;
  // 连续32个线程按 0 8 16 24 32... 排列  (tid%32) * 8
  int load_b_smem_n = (tid & 31) << 3;

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
  int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

  int comp_c_frag_m = wid & 1;
  int comp_c_frag_n = wid >> 1;

  int smem_a_base_addr = __cvta_generic_to_shared(smem_a[0]);
  int load_a_smem_addr_0 = smem_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
  int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);

  int smem_b_base_addr = __cvta_generic_to_shared(smem_b[0]);
  int load_b_smem_addr_0 = smem_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
  int load_b_smem_addr_1 = load_b_smem_addr_0 + (BN + BPAD) * sizeof(half);
  int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
  int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);

  
  for(int bk = 0; bk < K / BK; bk++){
    // FLOAT4(smem_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0), "l"(&a[load_a_gmem_addr]));
    // FLOAT4(smem_a[load_a_smem_m+1][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr+K]);
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_1), "l"(&a[load_a_gmem_addr + K]));
    // FLOAT4(smem_b[load_b_smem_k    ][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr        ]);
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr]));
    // FLOAT4(smem_b[load_b_smem_k + 1][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr +     N]);
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr + N]));
    // FLOAT4(smem_b[load_b_smem_k + 2][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + 2 * N]);
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2*N]));  //  
    // FLOAT4(smem_b[load_b_smem_k + 3][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + 3 * N]);
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3*N]));

    load_a_gmem_addr += BK;
    load_b_gmem_addr += BK * N;

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);

    __syncthreads();

    wmma::load_matrix_sync(frag_a[0][0], &smem_a[comp_c_frag_m * 64     ][ 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &smem_a[comp_c_frag_m * 64 + 16][ 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &smem_a[comp_c_frag_m * 64 + 32][ 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &smem_a[comp_c_frag_m * 64 + 48][ 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &smem_a[comp_c_frag_m * 64     ][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &smem_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &smem_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &smem_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &smem_b[ 0][comp_c_frag_n * 64     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &smem_b[ 0][comp_c_frag_n * 64 + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &smem_b[ 0][comp_c_frag_n * 64 + 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &smem_b[ 0][comp_c_frag_n * 64 + 48], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &smem_b[16][comp_c_frag_n * 64     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &smem_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &smem_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &smem_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
      }
    }

    __syncthreads();
  }

  int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
  int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
  int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

  for(int i = 0; i < 4; i++){
    for(int j = 0; j < 4; j++){
      wmma::store_matrix_sync(&c[store_c_gmem_addr + OFFSET(i,j,N) * 16], frag_c[i][j], N, wmma::mem_row_major);
    }
  }
}


template<int M, int N, int K, int BM, int BN, int BK>
__global__ void HGEMMV3(half *a, half *b, half *c){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = threadIdx.x;
  int wid = tid >> 5;

  const int APAD = 8;
  const int BPAD = 8;

  extern __shared__ half smem[];
  half *smem_a = smem;
  half *smem_b = smem + 2 * BM * (BK + APAD);
  int smem_a_db_offset = BM * (BK + APAD);
  int smem_b_db_offset = BK * (BN + BPAD);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
          wmma::fill_fragment(frag_c[i][j], 0.0);
      }
  }
  // 连续4个线程处理同一行; <<1表示一行4个线程会处理连续两行
  int load_a_smem_m = (tid >> 2) << 1;
  // 连续4个线程按 0 8 16 24 排列  (tid%4) * 8
  int load_a_smem_k = (tid & 3) << 3;
  // 连续32个线程处理同一行; <<2表示一行32个线程会处理连续四行
  int load_b_smem_k = (tid >> 5) << 2;
  // 连续32个线程按 0 8 16 24 32... 排列  (tid%32) * 8
  int load_b_smem_n = (tid & 31) << 3;

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
  int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

  int comp_c_frag_m = wid & 1;
  int comp_c_frag_n = wid >> 1;

  int smem_a_base_addr = __cvta_generic_to_shared(smem_a);
  int load_a_smem_addr_0 = smem_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
  int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);

  int smem_b_base_addr = __cvta_generic_to_shared(smem_b);
  int load_b_smem_addr_0 = smem_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
  int load_b_smem_addr_1 = load_b_smem_addr_0 + (BN + BPAD) * sizeof(half);
  int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
  int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);


  {
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_a_smem_addr_0), "l"(&a[load_a_gmem_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_a_smem_addr_1), "l"(&a[load_a_gmem_addr +     K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * N]));

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);

    __syncthreads();
  }
  
  for(int bk = 1; bk < K / BK; bk++){
    int smem_sel = (bk & 1) ^ 1;
    int smem_sel_next = ((bk - 1) & 1) ^ 1;

    load_a_gmem_addr += BK;
    load_b_gmem_addr += BK * N;
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + smem_sel_next * (int)sizeof(half) * smem_a_db_offset), "l"(&a[load_a_gmem_addr]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_1 + smem_sel_next * (int)sizeof(half) * smem_a_db_offset), "l"(&a[load_a_gmem_addr + K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * (int)sizeof(half) * smem_b_db_offset), "l"(&b[load_b_gmem_addr]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * (int)sizeof(half) * smem_b_db_offset), "l"(&b[load_b_gmem_addr + N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + smem_sel_next * (int)sizeof(half) * smem_b_db_offset), "l"(&b[load_b_gmem_addr + 2*N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + smem_sel_next * (int)sizeof(half) * smem_b_db_offset), "l"(&b[load_b_gmem_addr + 3*N]));

    wmma::load_matrix_sync(frag_a[0][0], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64) * (BN + BPAD) + 0], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 16) * (BN + BPAD) + 0], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 32) * (BN + BPAD) + 0], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 48) * (BN + BPAD) + 0], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64) * (BN + BPAD) + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 16) * (BN + BPAD) + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 32) * (BN + BPAD) + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 48) * (BN + BPAD) + 16], BN + BPAD);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
      }
    }

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);

    __syncthreads();
  }

  int smem_sel = ((K / BK) & 1) ^ 1;

  wmma::load_matrix_sync(frag_a[0][0], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 0], BK + APAD);
  wmma::load_matrix_sync(frag_a[0][1], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 0], BK + APAD);
  wmma::load_matrix_sync(frag_a[0][2], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 0], BK + APAD);
  wmma::load_matrix_sync(frag_a[0][3], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 0], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][0], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 16], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][1], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][2], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][3], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

  wmma::load_matrix_sync(frag_b[0][0], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64) * (BN + BPAD) + 0], BN + BPAD);
  wmma::load_matrix_sync(frag_b[0][1], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 16) * (BN + BPAD) + 0], BN + BPAD);
  wmma::load_matrix_sync(frag_b[0][2], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 32) * (BN + BPAD) + 0], BN + BPAD);
  wmma::load_matrix_sync(frag_b[0][3], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 48) * (BN + BPAD) + 0], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][0], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64) * (BN + BPAD) + 16], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][1], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 16) * (BN + BPAD) + 16], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][2], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 32) * (BN + BPAD) + 16], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][3], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 48) * (BN + BPAD) + 16], BN + BPAD);

  #pragma unroll
  for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
          wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
          wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
      }
  }

  int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
  int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
  int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

  #pragma unroll
  for(int i = 0; i < 4; i++){
    #pragma unroll
    for(int j = 0; j < 4; j++){
      wmma::store_matrix_sync(&c[store_c_gmem_addr + OFFSET(i,j,N) * 16], frag_c[i][j], N, wmma::mem_row_major);
    }
  }
}


template<int M, int N, int K, int BM, int BN, int BK>
__global__ void HGEMMV4(half *a, half *b, half *c){
  int bx = blockIdx.z * gridDim.x + blockIdx.x;
  int by = blockIdx.y;
  int tid = threadIdx.x;
  int wid = tid >> 5;

  const int APAD = 8;
  const int BPAD = 8;

  extern __shared__ half smem[];
  half *smem_a = smem;
  half *smem_b = smem + 2 * BM * (BK + APAD);
  int smem_a_db_offset = BM * (BK + APAD);
  int smem_b_db_offset = BK * (BN + BPAD);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
          wmma::fill_fragment(frag_c[i][j], 0.0);
      }
  }
  // 连续4个线程处理同一行; <<1表示一行4个线程会处理连续两行
  int load_a_smem_m = (tid >> 2) << 1;
  // 连续4个线程按 0 8 16 24 排列  (tid%4) * 8
  int load_a_smem_k = (tid & 3) << 3;
  // 连续32个线程处理同一行; <<2表示一行32个线程会处理连续四行
  int load_b_smem_k = (tid >> 5) << 2;
  // 连续32个线程按 0 8 16 24 32... 排列  (tid%32) * 8
  int load_b_smem_n = (tid & 31) << 3;

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
  int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

  int comp_c_frag_m = wid & 1;
  int comp_c_frag_n = wid >> 1;

  int smem_a_base_addr = __cvta_generic_to_shared(smem_a);
  int load_a_smem_addr_0 = smem_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
  int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);

  int smem_b_base_addr = __cvta_generic_to_shared(smem_b);
  int load_b_smem_addr_0 = smem_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
  int load_b_smem_addr_1 = load_b_smem_addr_0 + (BN + BPAD) * sizeof(half);
  int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
  int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);


  {
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_a_smem_addr_0), "l"(&a[load_a_gmem_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_a_smem_addr_1), "l"(&a[load_a_gmem_addr +     K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * N]));

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);

    __syncthreads();
  }
  
  for(int bk = 1; bk < K / BK; bk++){
    int smem_sel = (bk & 1) ^ 1;
    int smem_sel_next = ((bk - 1) & 1) ^ 1;

    load_a_gmem_addr += BK;
    load_b_gmem_addr += BK * N;
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + smem_sel_next * (int)sizeof(half) * smem_a_db_offset), "l"(&a[load_a_gmem_addr]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_1 + smem_sel_next * (int)sizeof(half) * smem_a_db_offset), "l"(&a[load_a_gmem_addr + K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * (int)sizeof(half) * smem_b_db_offset), "l"(&b[load_b_gmem_addr]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * (int)sizeof(half) * smem_b_db_offset), "l"(&b[load_b_gmem_addr + N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + smem_sel_next * (int)sizeof(half) * smem_b_db_offset), "l"(&b[load_b_gmem_addr + 2*N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + smem_sel_next * (int)sizeof(half) * smem_b_db_offset), "l"(&b[load_b_gmem_addr + 3*N]));

    wmma::load_matrix_sync(frag_a[0][0], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64) * (BN + BPAD) + 0], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 16) * (BN + BPAD) + 0], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 32) * (BN + BPAD) + 0], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 48) * (BN + BPAD) + 0], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64) * (BN + BPAD) + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 16) * (BN + BPAD) + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 32) * (BN + BPAD) + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 48) * (BN + BPAD) + 16], BN + BPAD);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
      }
    }

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);

    __syncthreads();
  }

  int smem_sel = ((K / BK) & 1) ^ 1;

  wmma::load_matrix_sync(frag_a[0][0], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 0], BK + APAD);
  wmma::load_matrix_sync(frag_a[0][1], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 0], BK + APAD);
  wmma::load_matrix_sync(frag_a[0][2], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 0], BK + APAD);
  wmma::load_matrix_sync(frag_a[0][3], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 0], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][0], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 16], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][1], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][2], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][3], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

  wmma::load_matrix_sync(frag_b[0][0], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64) * (BN + BPAD) + 0], BN + BPAD);
  wmma::load_matrix_sync(frag_b[0][1], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 16) * (BN + BPAD) + 0], BN + BPAD);
  wmma::load_matrix_sync(frag_b[0][2], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 32) * (BN + BPAD) + 0], BN + BPAD);
  wmma::load_matrix_sync(frag_b[0][3], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 48) * (BN + BPAD) + 0], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][0], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64) * (BN + BPAD) + 16], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][1], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 16) * (BN + BPAD) + 16], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][2], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 32) * (BN + BPAD) + 16], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][3], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 48) * (BN + BPAD) + 16], BN + BPAD);

  #pragma unroll
  for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
          wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
          wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
      }
  }

  int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
  int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
  int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

  #pragma unroll
  for(int i = 0; i < 4; i++){
    #pragma unroll
    for(int j = 0; j < 4; j++){
      wmma::store_matrix_sync(&c[store_c_gmem_addr + OFFSET(i,j,N) * 16], frag_c[i][j], N, wmma::mem_row_major);
    }
  }
}


template<int M, int N, int K, int BM, int BN, int BK>
__global__ void HGEMMV5(half *a, half *b, half *c){
  int bx = blockIdx.z * gridDim.x + blockIdx.x;
  int by = blockIdx.y;
  int tid = threadIdx.x;
  int wid = tid >> 5;

  const int APAD = 8;
  const int BPAD = 8;

  extern __shared__ half smem[];
  half *smem_a = smem;
  half *smem_b = smem + 2 * BM * (BK + APAD);
  int smem_a_db_offset = BM * (BK + APAD);
  int smem_b_db_offset = BK * (BN + BPAD);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
          wmma::fill_fragment(frag_c[i][j], 0.0);
      }
  }
  // 连续4个线程处理同一行; <<1表示一行4个线程会处理连续两行
  int load_a_smem_m = (tid >> 2) << 1;
  // 连续4个线程按 0 8 16 24 排列  (tid%4) * 8
  int load_a_smem_k = (tid & 3) << 3;
  // 连续32个线程处理同一行; <<2表示一行32个线程会处理连续四行
  int load_b_smem_k = (tid >> 5) << 2;
  // 连续32个线程按 0 8 16 24 32... 排列  (tid%32) * 8
  int load_b_smem_n = (tid & 31) << 3;

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
  int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

  int comp_c_frag_m = wid & 1;
  int comp_c_frag_n = wid >> 1;

  int smem_a_base_addr = __cvta_generic_to_shared(smem_a);
  int load_a_smem_addr_0 = smem_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
  int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);

  int smem_b_base_addr = __cvta_generic_to_shared(smem_b);
  int load_b_smem_addr_0 = smem_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
  int load_b_smem_addr_1 = load_b_smem_addr_0 + (BN + BPAD) * sizeof(half);
  int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
  int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);


  {
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_a_smem_addr_0), "l"(&a[load_a_gmem_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_a_smem_addr_1), "l"(&a[load_a_gmem_addr +     K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * N]));

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);

    __syncthreads();
  }
  
  #pragma unroll 32
  for(int bk = 1; bk < K / BK; bk++){
    int smem_sel = (bk & 1) ^ 1;
    int smem_sel_next = ((bk - 1) & 1) ^ 1;

    load_a_gmem_addr += BK;
    load_b_gmem_addr += BK * N;
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + smem_sel_next * (int)sizeof(half) * smem_a_db_offset), "l"(&a[load_a_gmem_addr]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_1 + smem_sel_next * (int)sizeof(half) * smem_a_db_offset), "l"(&a[load_a_gmem_addr + K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * (int)sizeof(half) * smem_b_db_offset), "l"(&b[load_b_gmem_addr]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * (int)sizeof(half) * smem_b_db_offset), "l"(&b[load_b_gmem_addr + N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + smem_sel_next * (int)sizeof(half) * smem_b_db_offset), "l"(&b[load_b_gmem_addr + 2*N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + smem_sel_next * (int)sizeof(half) * smem_b_db_offset), "l"(&b[load_b_gmem_addr + 3*N]));

    wmma::load_matrix_sync(frag_a[0][0], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64) * (BN + BPAD) + 0], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 16) * (BN + BPAD) + 0], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 32) * (BN + BPAD) + 0], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 48) * (BN + BPAD) + 0], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64) * (BN + BPAD) + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 16) * (BN + BPAD) + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 32) * (BN + BPAD) + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 48) * (BN + BPAD) + 16], BN + BPAD);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
      }
    }

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);

    __syncthreads();
  }

  int smem_sel = ((K / BK) & 1) ^ 1;

  wmma::load_matrix_sync(frag_a[0][0], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 0], BK + APAD);
  wmma::load_matrix_sync(frag_a[0][1], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 0], BK + APAD);
  wmma::load_matrix_sync(frag_a[0][2], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 0], BK + APAD);
  wmma::load_matrix_sync(frag_a[0][3], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 0], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][0], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 16], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][1], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][2], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][3], &smem_a[smem_sel * smem_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

  wmma::load_matrix_sync(frag_b[0][0], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64) * (BN + BPAD) + 0], BN + BPAD);
  wmma::load_matrix_sync(frag_b[0][1], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 16) * (BN + BPAD) + 0], BN + BPAD);
  wmma::load_matrix_sync(frag_b[0][2], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 32) * (BN + BPAD) + 0], BN + BPAD);
  wmma::load_matrix_sync(frag_b[0][3], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 48) * (BN + BPAD) + 0], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][0], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64) * (BN + BPAD) + 16], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][1], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 16) * (BN + BPAD) + 16], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][2], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 32) * (BN + BPAD) + 16], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][3], &smem_b[smem_sel * smem_b_db_offset + (comp_c_frag_m * 64 + 48) * (BN + BPAD) + 16], BN + BPAD);

  #pragma unroll
  for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
          wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
          wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
      }
  }

  int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
  int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
  int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

  #pragma unroll
  for(int i = 0; i < 4; i++){
    #pragma unroll
    for(int j = 0; j < 4; j++){
      wmma::store_matrix_sync(&c[store_c_gmem_addr + OFFSET(i,j,N) * 16], frag_c[i][j], N, wmma::mem_row_major);
    }
  }
}



template<int M, int N, int K, int BM, int BN, int BK>
void launch_myHGEMM(int version, half *d_a, half *d_b, half *d_c){
  unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
  dim3 block(256);
  dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
  const unsigned int NSPLIT = 4096;
  unsigned int split_num = (N + NSPLIT - 1) / NSPLIT;
  unsigned int BX = (N+BN-1)/BN;

  switch (version) {
    case 1:
      HGEMMV1<M, N, K, BM, BN, BK><<<grid, block>>>(d_a, d_b, d_c);
      break;
    case 2:
      HGEMMV2<M, N, K, BM, BN, BK><<<grid, block>>>(d_a, d_b, d_c);
      break;
    case 3:
      // gemmv3 要用动态分配是因为 静态分配报错太多smem被使用。因为默认分配的 l1cache:smem 太小，得主动放大。
      HGEMMV3<M, N, K, BM, BN, BK><<<grid, block, dsmem>>>(d_a, d_b, d_c);
      break;
    case 4:
      grid = {(BX + split_num - 1) / split_num, (M+BM-1)/BM, split_num};
      // A100上限164KB/SM 默认 48KB/SM 改成96KB/SM
      cudaFuncSetAttribute(HGEMMV4<M, N, K, BM, BN, BK>, cudaFuncAttributePreferredSharedMemoryCarveout, 98304);
      HGEMMV4<M, N, K, BM, BN, BK><<<grid, block, dsmem>>>(d_a, d_b, d_c);
      break;
    case 5:
      grid = {(BX + split_num - 1) / split_num, (M+BM-1)/BM, split_num};
      // A100上限164KB/SM 默认 48KB/SM 改成96KB/SM
      cudaFuncSetAttribute(HGEMMV5<M, N, K, BM, BN, BK>, cudaFuncAttributePreferredSharedMemoryCarveout, 98304);
      HGEMMV5<M, N, K, BM, BN, BK><<<grid, block, dsmem>>>(d_a, d_b, d_c);
      break;
    default:
      throw std::runtime_error("version error");
  }
}


int main(){
  cudaSetDevice(1); 
  // int device;
  // cudaGetDevice(&device);
  // cudaDeviceProp prop;
  // cudaGetDeviceProperties(&prop, device);
  // std::cout << "每个block的共享内存大小：" << prop.sharedMemPerBlock << std::endl;

  // warmup();

  // const int M = 256;
  // const int N = 256;
  // const int K = 1024;
  const int M = 1024;
  const int N = 1024;
  const int K = 1024;

  const int BM = 128;
  const int BN = 256;
  const int BK = 32;

  size_t size_a = M * K * sizeof(half);
  size_t size_b = K * N * sizeof(half);
  size_t size_c = M * N * sizeof(half);

  half *h_a, *h_b, *d_a, *d_b;
  half *h_c, *d_c, *h_d_c;
  h_a = (half *)malloc(size_a);
  h_b = (half *)malloc(size_b);
  h_c = (half *)malloc(size_c);
  CHECK(cudaMalloc(&d_a, size_a));
  CHECK(cudaMalloc(&d_b, size_b));
  CHECK(cudaMalloc(&d_c, size_c));
  h_d_c = (half *)malloc(size_c);

  assignData(M, N, K, h_a, h_b, h_c, h_d_c);

  CHECK(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));

  // /// cpuGemm
  // cpuGemm<M, N, K>(h_a, h_b, h_d_c);

  /// cublas
  const half alpha = 1.0f;
  const half beta = 0.0f;
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost));
  cublasDestroy(handle);
  
  /// my
  int version = 5;
  launch_myHGEMM<M, N, K, BM, BN, BK>(version, d_a, d_b, d_c);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost));

  /// valid
  // print(M, K, h_a);
  valid(M, N, h_d_c, h_c);

  free(h_a); free(h_b); free(h_c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); free(h_d_c);

}


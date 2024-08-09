#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cublas_v2.h"
#include <mma.h>

#include <random>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <cstdio>
#include <cassert>

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
  int cnt = 10;
  for(int m = 0; m < M; m++){
    for(int n = 0; n < N; n++){
      float my_c_ele = __half2float(my_c[m*N+n]);
      float ref_c_ele = __half2float(ref_c[m*N+n]);
      float diff = std::abs(my_c_ele - ref_c_ele);  
      if (diff / ref_c_ele > 0.01 && cnt >= 0) {  
        succ = false;
        printf("Error: [%d, %d],     my_c_ele: %f,     ref_c_ele: %f\n", m, n, my_c_ele, ref_c_ele);
        cnt--;
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
__global__ void my_HGEMMV5(half *a, half *b, half *c){
  int tid = threadIdx.x;
  int bix = blockIdx.z * gridDim.x + blockIdx.x;
  int biy = blockIdx.y;
  int wid = tid >> 5;

  const int APAD = 8;
  const int BPAD = 8;
  extern __shared__  half smem[];
  half *smem_a = smem;
  half *smem_b = smem + 2 * BM * (BK + APAD);
  int smem_a_db_offset = BM * (BK + APAD);
  int smem_b_db_offset = BN * (BN + BPAD);
  
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
  
  int lane_id = tid & 31;
  int sts_a_lane_row_id = (lane_id / 4);
  int sts_a_warp_row_id = wid * 16;
  int sts_a_row = sts_a_warp_row_id + (sts_a_lane_row_id % 2) * 8 + (sts_a_lane_row_id / 2) * 2;
  
  int sts_a_col = (tid << 3) & 31;
  int sts_b_row = ((tid << 3) >> 8) << 2;
  int sts_b_col = (tid << 3) & 255;

  int ldg_a_row = biy * BM + sts_a_row;
  int ldg_a_col = sts_a_col;
  int ldg_a_addr = OFFSET(ldg_a_row, ldg_a_col, K);
  int ldg_b_row = sts_b_row;
  int ldg_b_col = bix * BN + sts_b_col;
  int ldg_b_addr = OFFSET(ldg_b_row, ldg_b_col, N);

  int lds_a_row = (wid >> 2) << 4;
  int lds_b_col = (wid & 3) << 4;

  int smem_a_base_addr = __cvta_generic_to_shared(smem_a);
  int smem_b_base_addr = __cvta_generic_to_shared(smem_b);

  int sts_a_addr_0 = smem_a_base_addr + OFFSET(sts_a_row, sts_a_col, BK + APAD) * sizeof(half);
  int sts_a_addr_1 = sts_a_addr_0 + (BK + APAD) * sizeof(half);
  int sts_b_addr_0 = smem_b_base_addr + OFFSET(sts_b_row, sts_b_col, BN + BPAD) * sizeof(half);
  int sts_b_addr_1 = sts_b_addr_0 + (BN + BPAD) * sizeof(half);
  int sts_b_addr_2 = sts_b_addr_1 + (BN + BPAD) * sizeof(half);
  int sts_b_addr_3 = sts_b_addr_2 + (BN + BPAD) * sizeof(half);

  {
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(sts_a_addr_0), "l"(&a[ldg_a_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(sts_a_addr_1), "l"(&a[ldg_a_addr +     K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(sts_b_addr_0), "l"(&b[ldg_b_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(sts_b_addr_1), "l"(&b[ldg_b_addr +     N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(sts_b_addr_2), "l"(&b[ldg_b_addr + 2 * N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(sts_b_addr_3), "l"(&b[ldg_b_addr + 3 * N]));

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);

    __syncthreads();
  }

  #pragma unroll 32
  for(int k = 1; k < K / BK; k++){
    int smem_sel = (k-1) & 1;
    int smem_next = k & 1;

    ldg_a_addr += BK;
    ldg_b_addr += N * BK;

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : 
          : "r"(sts_a_addr_0 + smem_next * smem_a_db_offset * (int)sizeof(half)), "l"(&a[ldg_a_addr    ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : 
          : "r"(sts_a_addr_1 + smem_next * smem_a_db_offset * (int)sizeof(half)), "l"(&a[ldg_a_addr + K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
          : "r"(sts_b_addr_0 + smem_next * smem_b_db_offset * (int)sizeof(half)), "l"(&b[ldg_b_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : 
          : "r"(sts_b_addr_1 + smem_next * smem_b_db_offset * (int)sizeof(half)), "l"(&b[ldg_b_addr +     N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : 
          : "r"(sts_b_addr_2 + smem_next * smem_b_db_offset * (int)sizeof(half)), "l"(&b[ldg_b_addr + 2 * N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : 
          : "r"(sts_b_addr_3 + smem_next * smem_b_db_offset * (int)sizeof(half)), "l"(&b[ldg_b_addr + 3 * N]));
        
    wmma::load_matrix_sync(frag_a[0][0], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row     ) * (BK + APAD) + 0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32  ) * (BK + APAD) + 0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*2) * (BK + APAD) + 0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*3) * (BK + APAD) + 0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row     ) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32  ) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*2) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*3) * (BK + APAD) + 16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &smem_b[smem_sel * smem_b_db_offset + lds_b_col     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &smem_b[smem_sel * smem_b_db_offset + lds_b_col+64  ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &smem_b[smem_sel * smem_b_db_offset + lds_b_col+64*2], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &smem_b[smem_sel * smem_b_db_offset + lds_b_col+64*3], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col+64  ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col+64*2], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col+64*3], BK + APAD);

    #pragma unroll
    for(int i = 0; i < 4; i++){
      #pragma unroll
      for(int j = 0; j < 4; j++){
        wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
      }
    }

    asm ("cp.async.wait_all;\n" : : );
    __syncthreads();
  }

  int smem_sel = (K/BK - 1) & 1;
  wmma::load_matrix_sync(frag_a[0][0], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row     ) * (BK + APAD) + 0 ], BK + APAD);
  wmma::load_matrix_sync(frag_a[0][1], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32  ) * (BK + APAD) + 0 ], BK + APAD);
  wmma::load_matrix_sync(frag_a[0][2], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*2) * (BK + APAD) + 0 ], BK + APAD);
  wmma::load_matrix_sync(frag_a[0][3], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*3) * (BK + APAD) + 0 ], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][0], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row     ) * (BK + APAD) + 16], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][1], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32  ) * (BK + APAD) + 16], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][2], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*2) * (BK + APAD) + 16], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][3], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*3) * (BK + APAD) + 16], BK + APAD);

  wmma::load_matrix_sync(frag_b[0][0], &smem_b[smem_sel * smem_b_db_offset + lds_b_col     ], BN + BPAD);
  wmma::load_matrix_sync(frag_b[0][1], &smem_b[smem_sel * smem_b_db_offset + lds_b_col+64  ], BN + BPAD);
  wmma::load_matrix_sync(frag_b[0][2], &smem_b[smem_sel * smem_b_db_offset + lds_b_col+64*2], BN + BPAD);
  wmma::load_matrix_sync(frag_b[0][3], &smem_b[smem_sel * smem_b_db_offset + lds_b_col+64*3], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][0], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col     ], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][1], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col+64  ], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][2], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col+64*2], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][3], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col+64*3], BK + APAD);

  #pragma unroll
  for(int i = 0; i < 4; i++){
    #pragma unroll
    for(int j = 0; j < 4; j++){
      wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
      wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
    }
  }

  int stg_c_row = biy * BM + lds_a_row;
  int stg_c_col = bix * BN + lds_b_col;
  int stg_c_addr = OFFSET(stg_c_row, stg_c_col, N);
  #pragma unroll
  for(int i = 0; i < 4; i++){
    #pragma unroll
    for(int j = 0; j < 4; j++){
      wmma::store_matrix_sync(&c[stg_c_addr + OFFSET(32*i, 64*j, N)], frag_c[i][j], N, wmma::mem_row_major);
    }
  }
}

// double buffer
template<int M, int N, int K, int BM, int BN, int BK>
__global__ void my_HGEMMV4(half *a, half *b, half *c){
  int tid = threadIdx.x;
  int bix = blockIdx.x;
  int biy = blockIdx.y;
  int wid = tid >> 5;

  const int APAD = 8;
  const int BPAD = 8;
  extern __shared__  half smem[];
  half *smem_a = smem;
  half *smem_b = smem + 2 * BM * (BK + APAD);
  int smem_a_db_offset = BM * (BK + APAD);
  int smem_b_db_offset = BN * (BN + BPAD);
  
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
  
  int lane_id = tid & 31;
  int sts_a_lane_row_id = (lane_id / 4);
  int sts_a_warp_row_id = wid * 16;
  int sts_a_row = sts_a_warp_row_id + (sts_a_lane_row_id % 2) * 8 + (sts_a_lane_row_id / 2) * 2;
  
  int sts_a_col = (tid << 3) & 31;
  int sts_b_row = ((tid << 3) >> 8) << 2;
  int sts_b_col = (tid << 3) & 255;

  int ldg_a_row = biy * BM + sts_a_row;
  int ldg_a_col = sts_a_col;
  int ldg_a_addr = OFFSET(ldg_a_row, ldg_a_col, K);
  int ldg_b_row = sts_b_row;
  int ldg_b_col = bix * BN + sts_b_col;
  int ldg_b_addr = OFFSET(ldg_b_row, ldg_b_col, N);

  int lds_a_row = (wid >> 2) << 4;
  int lds_b_col = (wid & 3) << 4;

  int smem_a_base_addr = __cvta_generic_to_shared(smem_a);
  int smem_b_base_addr = __cvta_generic_to_shared(smem_b);

  int sts_a_addr_0 = smem_a_base_addr + OFFSET(sts_a_row, sts_a_col, BK + APAD) * sizeof(half);
  int sts_a_addr_1 = sts_a_addr_0 + (BK + APAD) * sizeof(half);
  int sts_b_addr_0 = smem_b_base_addr + OFFSET(sts_b_row, sts_b_col, BN + BPAD) * sizeof(half);
  int sts_b_addr_1 = sts_b_addr_0 + (BN + BPAD) * sizeof(half);
  int sts_b_addr_2 = sts_b_addr_1 + (BN + BPAD) * sizeof(half);
  int sts_b_addr_3 = sts_b_addr_2 + (BN + BPAD) * sizeof(half);

  {
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(sts_a_addr_0), "l"(&a[ldg_a_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(sts_a_addr_1), "l"(&a[ldg_a_addr +     K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(sts_b_addr_0), "l"(&b[ldg_b_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(sts_b_addr_1), "l"(&b[ldg_b_addr +     N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(sts_b_addr_2), "l"(&b[ldg_b_addr + 2 * N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "r"(sts_b_addr_3), "l"(&b[ldg_b_addr + 3 * N]));

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);

    __syncthreads();
  }

  #pragma unroll 32
  for(int k = 1; k < K / BK; k++){
    int smem_sel = (k-1) & 1;
    int smem_next = k & 1;

    ldg_a_addr += BK;
    ldg_b_addr += N * BK;

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : 
          : "r"(sts_a_addr_0 + smem_next * smem_a_db_offset * (int)sizeof(half)), "l"(&a[ldg_a_addr    ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : 
          : "r"(sts_a_addr_1 + smem_next * smem_a_db_offset * (int)sizeof(half)), "l"(&a[ldg_a_addr + K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
          : "r"(sts_b_addr_0 + smem_next * smem_b_db_offset * (int)sizeof(half)), "l"(&b[ldg_b_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : 
          : "r"(sts_b_addr_1 + smem_next * smem_b_db_offset * (int)sizeof(half)), "l"(&b[ldg_b_addr +     N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : 
          : "r"(sts_b_addr_2 + smem_next * smem_b_db_offset * (int)sizeof(half)), "l"(&b[ldg_b_addr + 2 * N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : 
          : "r"(sts_b_addr_3 + smem_next * smem_b_db_offset * (int)sizeof(half)), "l"(&b[ldg_b_addr + 3 * N]));
        
    wmma::load_matrix_sync(frag_a[0][0], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row     ) * (BK + APAD) + 0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32  ) * (BK + APAD) + 0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*2) * (BK + APAD) + 0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*3) * (BK + APAD) + 0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row     ) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32  ) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*2) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*3) * (BK + APAD) + 16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &smem_b[smem_sel * smem_b_db_offset + lds_b_col     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &smem_b[smem_sel * smem_b_db_offset + lds_b_col+64  ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &smem_b[smem_sel * smem_b_db_offset + lds_b_col+64*2], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &smem_b[smem_sel * smem_b_db_offset + lds_b_col+64*3], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col+64  ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col+64*2], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col+64*3], BK + APAD);

    #pragma unroll
    for(int i = 0; i < 4; i++){
      #pragma unroll
      for(int j = 0; j < 4; j++){
        wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
      }
    }

    asm ("cp.async.wait_all;\n" : : );
    __syncthreads();
  }

  int smem_sel = (K/BK - 1) & 1;
  wmma::load_matrix_sync(frag_a[0][0], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row     ) * (BK + APAD) + 0 ], BK + APAD);
  wmma::load_matrix_sync(frag_a[0][1], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32  ) * (BK + APAD) + 0 ], BK + APAD);
  wmma::load_matrix_sync(frag_a[0][2], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*2) * (BK + APAD) + 0 ], BK + APAD);
  wmma::load_matrix_sync(frag_a[0][3], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*3) * (BK + APAD) + 0 ], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][0], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row     ) * (BK + APAD) + 16], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][1], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32  ) * (BK + APAD) + 16], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][2], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*2) * (BK + APAD) + 16], BK + APAD);
  wmma::load_matrix_sync(frag_a[1][3], &smem_a[smem_sel * smem_a_db_offset + (lds_a_row+32*3) * (BK + APAD) + 16], BK + APAD);

  wmma::load_matrix_sync(frag_b[0][0], &smem_b[smem_sel * smem_b_db_offset + lds_b_col     ], BN + BPAD);
  wmma::load_matrix_sync(frag_b[0][1], &smem_b[smem_sel * smem_b_db_offset + lds_b_col+64  ], BN + BPAD);
  wmma::load_matrix_sync(frag_b[0][2], &smem_b[smem_sel * smem_b_db_offset + lds_b_col+64*2], BN + BPAD);
  wmma::load_matrix_sync(frag_b[0][3], &smem_b[smem_sel * smem_b_db_offset + lds_b_col+64*3], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][0], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col     ], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][1], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col+64  ], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][2], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col+64*2], BN + BPAD);
  wmma::load_matrix_sync(frag_b[1][3], &smem_b[smem_sel * smem_b_db_offset + 16 * (BN + BPAD) + lds_b_col+64*3], BK + APAD);

  #pragma unroll
  for(int i = 0; i < 4; i++){
    #pragma unroll
    for(int j = 0; j < 4; j++){
      wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
      wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
    }
  }

  int stg_c_row = biy * BM + lds_a_row;
  int stg_c_col = bix * BN + lds_b_col;
  int stg_c_addr = OFFSET(stg_c_row, stg_c_col, N);
  #pragma unroll
  for(int i = 0; i < 4; i++){
    #pragma unroll
    for(int j = 0; j < 4; j++){
      wmma::store_matrix_sync(&c[stg_c_addr + OFFSET(32*i, 64*j, N)], frag_c[i][j], N, wmma::mem_row_major);
    }
  }
}

// ldgsts
template<int M, int N, int K, int BM, int BN, int BK>
__global__ void my_HGEMMV3(half *a, half *b, half *c){
  int tid = threadIdx.x;
  int bix = blockIdx.x;
  int biy = blockIdx.y;
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
  
  int lane_id = tid & 31;
  int sts_a_lane_row_id = (lane_id / 4);
  int sts_a_warp_row_id = wid * 16;
  int sts_a_row = sts_a_warp_row_id + (sts_a_lane_row_id % 2) * 8 + (sts_a_lane_row_id / 2) * 2;
  
  int sts_a_col = (tid << 3) & 31;
  int sts_b_row = ((tid << 3) >> 8) << 2;
  int sts_b_col = (tid << 3) & 255;

  int ldg_a_row = biy * BM + sts_a_row;
  int ldg_a_col = sts_a_col;
  int ldg_a_addr = OFFSET(ldg_a_row, ldg_a_col, K);
  int ldg_b_row = sts_b_row;
  int ldg_b_col = bix * BN + sts_b_col;
  int ldg_b_addr = OFFSET(ldg_b_row, ldg_b_col, N);

  int lds_a_row = (wid >> 2) << 4;
  int lds_b_col = (wid & 3) << 4;

  // int smem_a_base_addr = __cvta_generic_to_shared(smem_a[0]);
  uint32_t smem_a_base_addr;
  asm volatile (
    "{.reg .u64 u64addr;\n"
    " cvta.to.shared.u64 u64addr, %1;\n"
    " cvt.u32.u64 %0, u64addr;}\n"
    : "=r"(smem_a_base_addr)
    : "l"(smem_a[0])
  );
  // int smem_b_base_addr = __cvta_generic_to_shared(smem_b[0]);
  uint32_t smem_b_base_addr;
  asm volatile (
    "{.reg .u64 u64addr;\n"
    " cvta.to.shared.u64 u64addr, %1;\n"
    " cvt.u32.u64 %0, u64addr;}\n"
    : "=r"(smem_b_base_addr)
    : "l"(smem_b[0])
  );

  int sts_a_addr_0 = smem_a_base_addr + OFFSET(sts_a_row, sts_a_col, BK + APAD) * sizeof(half);
  int sts_a_addr_1 = sts_a_addr_0 + (BK + APAD) * sizeof(half);
  int sts_b_addr_0 = smem_b_base_addr + OFFSET(sts_b_row, sts_b_col, BN + BPAD) * sizeof(half);
  int sts_b_addr_1 = sts_b_addr_0 + (BN + BPAD) * sizeof(half);
  int sts_b_addr_2 = sts_b_addr_1 + (BN + BPAD) * sizeof(half);
  int sts_b_addr_3 = sts_b_addr_2 + (BN + BPAD) * sizeof(half);

  for(int k = 0; k < K / BK; k++){
    // 输出 输入 内存 "r" = .u32 reg "l" = .u64 reg
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(sts_a_addr_0), "l"(&a[ldg_a_addr    ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(sts_a_addr_1), "l"(&a[ldg_a_addr + K]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(sts_b_addr_0), "l"(&b[ldg_b_addr        ]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(sts_b_addr_1), "l"(&b[ldg_b_addr +     N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(sts_b_addr_2), "l"(&b[ldg_b_addr + 2 * N]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(sts_b_addr_3), "l"(&b[ldg_b_addr + 3 * N]));

    ldg_a_addr += BK;
    ldg_b_addr += N * BK;

    asm ("cp.async.wait_all;\n" : : );
    __syncthreads();
    
    wmma::load_matrix_sync(frag_a[0][0], &smem_a[lds_a_row     ][0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &smem_a[lds_a_row+32  ][0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &smem_a[lds_a_row+32*2][0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &smem_a[lds_a_row+32*3][0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &smem_a[lds_a_row     ][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &smem_a[lds_a_row+32  ][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &smem_a[lds_a_row+32*2][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &smem_a[lds_a_row+32*3][16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &smem_b[0 ][lds_b_col     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &smem_b[0 ][lds_b_col+64  ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &smem_b[0 ][lds_b_col+64*2], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &smem_b[0 ][lds_b_col+64*3], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &smem_b[16][lds_b_col     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &smem_b[16][lds_b_col+64  ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &smem_b[16][lds_b_col+64*2], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &smem_b[16][lds_b_col+64*3], BN + BPAD);

    #pragma unroll
    for(int i = 0; i < 4; i++){
      #pragma unroll
      for(int j = 0; j < 4; j++){
        wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
      }
    }

    __syncthreads();
  }

  int stg_c_row = biy * BM + lds_a_row;
  int stg_c_col = bix * BN + lds_b_col;
  int stg_c_addr = OFFSET(stg_c_row, stg_c_col, N);
  #pragma unroll
  for(int i = 0; i < 4; i++){
    #pragma unroll
    for(int j = 0; j < 4; j++){
      wmma::store_matrix_sync(&c[stg_c_addr + OFFSET(32*i, 64*j, N)], frag_c[i][j], N, wmma::mem_row_major);
    }
  }
}

// 类似cute swizzle的smem写法
template<int M, int N, int K, int BM, int BN, int BK>
__global__ void my_HGEMMV2(half *a, half *b, half *c){
  int tid = threadIdx.x;
  int bix = blockIdx.x;
  int biy = blockIdx.y;
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

  // // << 3表示一个线程处理8个half, >> 5表示根据每行32个元素算行数, << 1表示希望连续处理两行
  // int sts_a_row = (tid >> 2) << 1;

  int lane_id = tid & 31;
  // 如果想让T0~T7是一个phase，不发生共享内存bank冲突
  int sts_a_lane_row_id = (lane_id / 4);
  int sts_a_warp_row_id = wid * 16;
  // sts_a_lane_row_id == 0 读第零行 && sts_a_lane_row_id == 1 读第八行
  // sts_a_lane_row_id == 2 读第二行 && sts_a_lane_row_id == 3 读第十行
  // sts_a_lane_row_id == 4 读第四行 && sts_a_lane_row_id == 5 读第十二行
  // sts_a_lane_row_id == 6 读第六行 && sts_a_lane_row_id == 7 读第十四行
  int sts_a_row = sts_a_warp_row_id + (sts_a_lane_row_id % 2) * 8 + (sts_a_lane_row_id / 2) * 2;
  
  int sts_a_col = (tid << 3) & 31;

  // << 3表示一个线程处理8个half, >> 8表示根据每行256个元素算行数, << 2表示希望连续处理四行
  int sts_b_row = ((tid << 3) >> 8) << 2;
  // << 3表示一个线程处理8个half, & 255表示线程的行内排布
  int sts_b_col = (tid << 3) & 255;

  int ldg_a_row = biy * BM + sts_a_row;
  int ldg_a_col = sts_a_col;
  int ldg_a_addr = OFFSET(ldg_a_row, ldg_a_col, K);
  int ldg_b_row = sts_b_row;
  int ldg_b_col = bix * BN + sts_b_col;
  int ldg_b_addr = OFFSET(ldg_b_row, ldg_b_col, N);

  // warp视角看寄存器读
  // >> 2表示每行4个warp，<< 4表示一个warp处理16行
  int lds_a_row = (wid >> 2) << 4;
  // & 31表示warp的行内排布， << 4表示一个warp处理16列
  int lds_b_col = (wid & 3) << 4;

  for(int k = 0; k < K / BK; k++){
    FLOAT4(smem_a[sts_a_row  ][sts_a_col]) = FLOAT4(a[ldg_a_addr    ]);
    FLOAT4(smem_a[sts_a_row+1][sts_a_col]) = FLOAT4(a[ldg_a_addr + K]);
    FLOAT4(smem_b[sts_b_row  ][sts_b_col]) = FLOAT4(b[ldg_b_addr        ]);
    FLOAT4(smem_b[sts_b_row+1][sts_b_col]) = FLOAT4(b[ldg_b_addr +     N]);
    FLOAT4(smem_b[sts_b_row+2][sts_b_col]) = FLOAT4(b[ldg_b_addr + 2 * N]);
    FLOAT4(smem_b[sts_b_row+3][sts_b_col]) = FLOAT4(b[ldg_b_addr + 3 * N]);
    ldg_a_addr += BK;
    ldg_b_addr += N * BK;

    __syncthreads();
    
    wmma::load_matrix_sync(frag_a[0][0], &smem_a[lds_a_row     ][0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &smem_a[lds_a_row+32  ][0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &smem_a[lds_a_row+32*2][0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &smem_a[lds_a_row+32*3][0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &smem_a[lds_a_row     ][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &smem_a[lds_a_row+32  ][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &smem_a[lds_a_row+32*2][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &smem_a[lds_a_row+32*3][16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &smem_b[0 ][lds_b_col     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &smem_b[0 ][lds_b_col+64  ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &smem_b[0 ][lds_b_col+64*2], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &smem_b[0 ][lds_b_col+64*3], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &smem_b[16][lds_b_col     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &smem_b[16][lds_b_col+64  ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &smem_b[16][lds_b_col+64*2], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &smem_b[16][lds_b_col+64*3], BN + BPAD);

    #pragma unroll
    for(int i = 0; i < 4; i++){
      #pragma unroll
      for(int j = 0; j < 4; j++){
        wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
      }
    }

    __syncthreads();
  }

  int stg_c_row = biy * BM + lds_a_row;
  int stg_c_col = bix * BN + lds_b_col;
  int stg_c_addr = OFFSET(stg_c_row, stg_c_col, N);
  #pragma unroll
  for(int i = 0; i < 4; i++){
    #pragma unroll
    for(int j = 0; j < 4; j++){
      wmma::store_matrix_sync(&c[stg_c_addr + OFFSET(32*i, 64*j, N)], frag_c[i][j], N, wmma::mem_row_major);
    }
  }
}

// 带PAD 防止共享内存bank conflict
template<int M, int N, int K, int BM, int BN, int BK>
__global__ void my_HGEMMV1(half *a, half *b, half *c){
  int tid = threadIdx.x;
  int bix = blockIdx.x;
  int biy = blockIdx.y;
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

  // << 3表示一个线程处理8个half, >> 5表示根据每行32个元素算行数, << 1表示希望连续处理两行
  int sts_a_row = ((tid << 3) >> 5) << 1;
  // << 3表示一个线程处理8个half, & 31表示线程的行内排布
  int sts_a_col = (tid << 3) & 31;
  // << 3表示一个线程处理8个half, >> 8表示根据每行256个元素算行数, << 2表示希望连续处理四行
  int sts_b_row = ((tid << 3) >> 8) << 2;
  // << 3表示一个线程处理8个half, & 255表示线程的行内排布
  int sts_b_col = (tid << 3) & 255;

  int ldg_a_row = biy * BM + sts_a_row;
  int ldg_a_col = sts_a_col;
  int ldg_a_addr = OFFSET(ldg_a_row, ldg_a_col, K);
  int ldg_b_row = sts_b_row;
  int ldg_b_col = bix * BN + sts_b_col;
  int ldg_b_addr = OFFSET(ldg_b_row, ldg_b_col, N);

  // warp视角看寄存器读
  // >> 2表示每行4个warp，<< 4表示一个warp处理16行
  int lds_a_row = (wid >> 2) << 4;
  // & 31表示warp的行内排布， << 4表示一个warp处理16列
  int lds_b_col = (wid & 3) << 4;

  for(int k = 0; k < K / BK; k++){
    FLOAT4(smem_a[sts_a_row  ][sts_a_col]) = FLOAT4(a[ldg_a_addr    ]);
    FLOAT4(smem_a[sts_a_row+1][sts_a_col]) = FLOAT4(a[ldg_a_addr + K]);
    FLOAT4(smem_b[sts_b_row  ][sts_b_col]) = FLOAT4(b[ldg_b_addr        ]);
    FLOAT4(smem_b[sts_b_row+1][sts_b_col]) = FLOAT4(b[ldg_b_addr +     N]);
    FLOAT4(smem_b[sts_b_row+2][sts_b_col]) = FLOAT4(b[ldg_b_addr + 2 * N]);
    FLOAT4(smem_b[sts_b_row+3][sts_b_col]) = FLOAT4(b[ldg_b_addr + 3 * N]);
    ldg_a_addr += BK;
    ldg_b_addr += N * BK;

    __syncthreads();
    
    wmma::load_matrix_sync(frag_a[0][0], &smem_a[lds_a_row     ][0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &smem_a[lds_a_row+32  ][0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &smem_a[lds_a_row+32*2][0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &smem_a[lds_a_row+32*3][0 ], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &smem_a[lds_a_row     ][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &smem_a[lds_a_row+32  ][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &smem_a[lds_a_row+32*2][16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &smem_a[lds_a_row+32*3][16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &smem_b[0 ][lds_b_col     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &smem_b[0 ][lds_b_col+64  ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &smem_b[0 ][lds_b_col+64*2], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &smem_b[0 ][lds_b_col+64*3], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &smem_b[16][lds_b_col     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &smem_b[16][lds_b_col+64  ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &smem_b[16][lds_b_col+64*2], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &smem_b[16][lds_b_col+64*3], BN + BPAD);

    #pragma unroll
    for(int i = 0; i < 4; i++){
      #pragma unroll
      for(int j = 0; j < 4; j++){
        wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
      }
    }

    __syncthreads();
  }

  int stg_c_row = biy * BM + lds_a_row;
  int stg_c_col = bix * BN + lds_b_col;
  int stg_c_addr = OFFSET(stg_c_row, stg_c_col, N);
  #pragma unroll
  for(int i = 0; i < 4; i++){
    #pragma unroll
    for(int j = 0; j < 4; j++){
      wmma::store_matrix_sync(&c[stg_c_addr + OFFSET(32*i, 64*j, N)], frag_c[i][j], N, wmma::mem_row_major);
    }
  }
}

template<int M, int N, int K, int BM, int BN, int BK>
__global__ void my_HGEMMV0(half *a, half *b, half *c){
  int tid = threadIdx.x;
  int bix = blockIdx.x;
  int biy = blockIdx.y;
  int wid = tid >> 5;

  __shared__ half smem_a[BM][BK];
  __shared__ half smem_b[BK][BN];

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

  // << 3表示一个线程处理8个half, >> 5表示根据每行32个元素算行数, << 1表示希望连续处理两行
  int sts_a_row = ((tid << 3) >> 5) << 1;
  // << 3表示一个线程处理8个half, & 31表示线程的行内排布
  int sts_a_col = (tid << 3) & 31;
  // << 3表示一个线程处理8个half, >> 8表示根据每行256个元素算行数, << 2表示希望连续处理四行
  int sts_b_row = ((tid << 3) >> 8) << 2;
  // << 3表示一个线程处理8个half, & 255表示线程的行内排布
  int sts_b_col = (tid << 3) & 255;

  int ldg_a_row = biy * BM + sts_a_row;
  int ldg_a_col = sts_a_col;
  int ldg_a_addr = OFFSET(ldg_a_row, ldg_a_col, K);
  int ldg_b_row = sts_b_row;
  int ldg_b_col = bix * BN + sts_b_col;
  int ldg_b_addr = OFFSET(ldg_b_row, ldg_b_col, N);

  // warp视角看寄存器读
  // >> 2表示每行4个warp，<< 4表示一个warp处理16行
  int lds_a_row = (wid >> 2) << 4;
  // & 31表示warp的行内排布， << 4表示一个warp处理16列
  int lds_b_col = (wid & 3) << 4;

  for(int k = 0; k < K / BK; k++){
    FLOAT4(smem_a[sts_a_row  ][sts_a_col]) = FLOAT4(a[ldg_a_addr    ]);
    FLOAT4(smem_a[sts_a_row+1][sts_a_col]) = FLOAT4(a[ldg_a_addr + K]);
    FLOAT4(smem_b[sts_b_row  ][sts_b_col]) = FLOAT4(b[ldg_b_addr        ]);
    FLOAT4(smem_b[sts_b_row+1][sts_b_col]) = FLOAT4(b[ldg_b_addr +     N]);
    FLOAT4(smem_b[sts_b_row+2][sts_b_col]) = FLOAT4(b[ldg_b_addr + 2 * N]);
    FLOAT4(smem_b[sts_b_row+3][sts_b_col]) = FLOAT4(b[ldg_b_addr + 3 * N]);
    ldg_a_addr += BK;
    ldg_b_addr += N * BK;

    __syncthreads();
    
    wmma::load_matrix_sync(frag_a[0][0], &smem_a[lds_a_row     ][0 ], BK);
    wmma::load_matrix_sync(frag_a[0][1], &smem_a[lds_a_row+32  ][0 ], BK);
    wmma::load_matrix_sync(frag_a[0][2], &smem_a[lds_a_row+32*2][0 ], BK);
    wmma::load_matrix_sync(frag_a[0][3], &smem_a[lds_a_row+32*3][0 ], BK);
    wmma::load_matrix_sync(frag_a[1][0], &smem_a[lds_a_row     ][16], BK);
    wmma::load_matrix_sync(frag_a[1][1], &smem_a[lds_a_row+32  ][16], BK);
    wmma::load_matrix_sync(frag_a[1][2], &smem_a[lds_a_row+32*2][16], BK);
    wmma::load_matrix_sync(frag_a[1][3], &smem_a[lds_a_row+32*3][16], BK);

    wmma::load_matrix_sync(frag_b[0][0], &smem_b[0 ][lds_b_col     ], BN);
    wmma::load_matrix_sync(frag_b[0][1], &smem_b[0 ][lds_b_col+64  ], BN);
    wmma::load_matrix_sync(frag_b[0][2], &smem_b[0 ][lds_b_col+64*2], BN);
    wmma::load_matrix_sync(frag_b[0][3], &smem_b[0 ][lds_b_col+64*3], BN);
    wmma::load_matrix_sync(frag_b[1][0], &smem_b[16][lds_b_col     ], BN);
    wmma::load_matrix_sync(frag_b[1][1], &smem_b[16][lds_b_col+64  ], BN);
    wmma::load_matrix_sync(frag_b[1][2], &smem_b[16][lds_b_col+64*2], BN);
    wmma::load_matrix_sync(frag_b[1][3], &smem_b[16][lds_b_col+64*3], BN);

    #pragma unroll
    for(int i = 0; i < 4; i++){
      #pragma unroll
      for(int j = 0; j < 4; j++){
        wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
        wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
      }
    }

    __syncthreads();
  }

  int stg_c_row = biy * BM + lds_a_row;
  int stg_c_col = bix * BN + lds_b_col;
  int stg_c_addr = OFFSET(stg_c_row, stg_c_col, N);
  #pragma unroll
  for(int i = 0; i < 4; i++){
    #pragma unroll
    for(int j = 0; j < 4; j++){
      wmma::store_matrix_sync(&c[stg_c_addr + OFFSET(32*i, 64*j, N)], frag_c[i][j], N, wmma::mem_row_major);
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
      my_HGEMMV1<M, N, K, BM, BN, BK><<<grid, block>>>(d_a, d_b, d_c);
      break;
    case 2:
      my_HGEMMV2<M, N, K, BM, BN, BK><<<grid, block>>>(d_a, d_b, d_c);
      break;
    case 3:
      my_HGEMMV3<M, N, K, BM, BN, BK><<<grid, block>>>(d_a, d_b, d_c);
      break;
    case 4:
      // gemmv4 要用动态分配是因为 静态分配报错太多smem被使用。因为默认分配的 l1cache:smem 太小，得主动放大。
      my_HGEMMV4<M, N, K, BM, BN, BK><<<grid, block, dsmem>>>(d_a, d_b, d_c);
      break;
    case 5:
      grid = {(BX + split_num - 1) / split_num, (M+BM-1)/BM, split_num};
      // A100上限164KB/SM 默认 48KB/SM 改成96KB/SM
      cudaFuncSetAttribute(my_HGEMMV5<M, N, K, BM, BN, BK>, cudaFuncAttributePreferredSharedMemoryCarveout, 98304);
      my_HGEMMV5<M, N, K, BM, BN, BK><<<grid, block, dsmem>>>(d_a, d_b, d_c);
      break;
    // case 6:
    //   grid = {(BX + split_num - 1) / split_num, (M+BM-1)/BM, split_num};
    //   // A100上限164KB/SM 默认 48KB/SM 改成96KB/SM
    //   cudaFuncSetAttribute(HGEMMV6<M, N, K, BM, BN, BK>, cudaFuncAttributePreferredSharedMemoryCarveout, 98304);
    //   my_HGEMMV6<M, N, K, BM, BN, BK><<<grid, block, dsmem>>>(d_a, d_b, d_c);
    //   break;
    default:
      throw std::runtime_error("version error");
  }
}


int main(){
  cudaSetDevice(4); 
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

  static_assert(M % BM == 0, "M % BM != 0");
  static_assert(N % BN == 0, "N % BN != 0");
  static_assert(K % BK == 0, "K % BK != 0");

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


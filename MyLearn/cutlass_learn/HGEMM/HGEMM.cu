#include <cstdio>
#include <iostream>
#include <string>
#include <chrono>
#include <ctime>
#include <vector>
#include "cublas_v2.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "utils.h"

#include "cutlass/gemm/device/gemm.h"

using DataType_ = half; 

cudaError_t ReferenceGemm(int M, int N, int K,
  DataType_ alpha, DataType_ const *A, int lda,
  DataType_ const *B, int ldb, DataType_ beta,
  DataType_ *C, int ldc);


cudaError_t LaunchCutlassHgemmNN(int M, int N, int K, 
                                    DataType_ alpha, 
                                    const DataType_* A, int lda, 
                                    const DataType_* B, int ldb,
                                    DataType_ beta,
                                    DataType_* C, int ldc
                                    ){
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<cutlass::half_t,        // Data-type of A matrix
                                                    ColumnMajor,        // Layout of A matrix
                                                    cutlass::half_t,        // Data-type of B matrix
                                                    ColumnMajor,        // Layout of B matrix
                                                    cutlass::half_t,        // Data-type of C matrix
                                                    ColumnMajor>;        // Layout of C matrix           
    CutlassGemm gemm_op;
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    CutlassGemm::Arguments args(problem_size,
                                {(cutlass::half_t *)A, lda},       // source A
                                {(cutlass::half_t *)B, ldb},       // source B
                                {(cutlass::half_t *)C, ldc},       // source C
                                {(cutlass::half_t *)C, ldc},       // dest D
                                {(cutlass::half_t)alpha, (cutlass::half_t)beta});

    // 执行gemm
    cutlass::Status status = gemm_op(args);
    
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

__global__ void InitMatrix_kernel(DataType_ *matrix, int rows, int cols, int seed){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.x + blockIdx.y * blockDim.y;
    if(i < rows && j < cols){
        int offset = i + j * rows;
        const int k = 16807;
        const int m = 16;
        DataType_ value =  __float2half(float(((offset + seed) * k % m) - m / 2));
        matrix[offset] = value;
    }
}

void InitMatrix(DataType_ *matrix, int rows, int cols, int seed){
    dim3 block(16, 16);
    dim3 grid((rows+block.x-1)/block.x, (cols+block.y-1)/block.y);
    InitMatrix_kernel<<<grid, block>>>(matrix, rows, cols, seed);
}

void AllocateMatrix(DataType_ **matrix, int rows, int cols, int seed=0){
    size_t size_bytes = sizeof(DataType_) * rows * cols;
    checkCudaErrors(cudaMalloc((void**)matrix, size_bytes));
    checkCudaErrors(cudaMemset(*matrix, 0, size_bytes));
    InitMatrix(*matrix, rows, cols, seed);
}

void TestCutlassHgemmNN(int M, int N, int K, DataType_ alpha, DataType_ beta){
    DataType_ *A;
    DataType_ *B;
    DataType_ *C_cutlass;
    DataType_ *C_reference;
    int lda = M;    // 列存 ld就是行数 
    int ldb = K;
    int ldc = M;

    AllocateMatrix(&A, M, K, 0);
    AllocateMatrix(&B, K, N, 17);
    AllocateMatrix(&C_cutlass, M, N, 101);
    AllocateMatrix(&C_reference, M, N, 101);
    size_t sizeof_C = sizeof(DataType_) * ldc * N;
    cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

    auto start = std::chrono::steady_clock::now();
    
    cudaError_t res = LaunchCutlassHgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);

    if (res != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: " << cudaGetErrorString(res) << std::endl;
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    cudaError_t res_ref = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);
    if (res_ref != cudaSuccess) {
        std::cerr << "Reference GEMM kernel failed: " << cudaGetErrorString(res_ref) << std::endl;
    }

    std::vector<float> host_cutlass(ldc * N, 0);
    std::vector<float> host_reference(ldc * N, 0);
    checkCudaErrors(cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost));

    if (host_cutlass != host_reference) {
        std::cerr << "CUTLASS results incorrect." << std::endl;
    }
    
    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);
}



int main(){
    // GEMM problem dimensions. 问题规模 m n k
    int problem[3] = { 128, 256, 4096 };
    // Scalars used for linear scaling the result of the matrix product. alpha和beta
    DataType_ scalars[2] = { 1, 0 };

    TestCutlassHgemmNN(problem[0], problem[1], problem[2], scalars[0], scalars[1]);
}

__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  DataType_ alpha,
  DataType_ const *A,
  int lda,
  DataType_ const *B,
  int ldb,
  DataType_ beta,
  DataType_ *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    DataType_ accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  DataType_ alpha,
  DataType_ const *A,
  int lda,
  DataType_ const *B,
  int ldb,
  DataType_ beta,
  DataType_ *C,
  int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}




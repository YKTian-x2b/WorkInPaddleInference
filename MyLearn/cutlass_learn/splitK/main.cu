#include "gemm_epilogue_util.h"
#include "gemm_epilogue_decl.h"
#include <random>
#include <ctime>
#include <vector>
#include <nvtx3/nvToolsExt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nvtx3/nvToolsExt.h>

// using DataType_ = __nv_bfloat16;
using DataType_ = half;

cutlass::Status matmul_add_sm80_fp16_15(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_81(const GemmEpilogueAllParams& params);
cutlass::Status matmul_add_sm80_fp16_99(const GemmEpilogueAllParams& params);

struct funcWithName{
    std::function<cutlass::Status(GemmEpilogueAllParams)> func;
    std::string funcName;
};

const std::vector<struct funcWithName> &all_func{
    {matmul_add_sm80_fp16_15, "matmul_add_sm80_fp16_15"},
    {matmul_add_sm80_fp16_81, "matmul_add_sm80_fp16_81"},
    {matmul_add_sm80_fp16_99, "matmul_add_sm80_fp16_99"},
};

typedef void (*func)(GemmEpilogueAllParams);

void InitMatrix(DataType_ *matrix, int rows, int cols);
void validRes(DataType_ *h_C_cublas, DataType_ *h_C_cutlass, size_t C_size);

void callCublasHGEMM(int M, int N, int K, DataType_* d_A, DataType_* d_B, 
                        DataType_* d_C_cublas, DataType_* h_C_cublas, size_t C_Bytes){
  cublasHandle_t handle;
  cublasCreate(&handle);
  DataType_ alpha = 1.0;
  DataType_ beta = 0.0;
  cublasStatus_t stat = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N);
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaMemcpy(h_C_cublas, d_C_cublas, C_Bytes, cudaMemcpyDeviceToHost));
}

void EnumParamList_MNK_Act(int M_, int N_, int K_, OpType op_type, std::string activation, float leaky_alpha_=0.01){
    int M = M_;
    int N = N_;
    int K = K_;
    // 行主序
    int lda = K, ldb = N, ldd = N;
    // int ldc_bias = 0;
    float leaky_alpha = leaky_alpha_;
    int sm_version = 80;
    GemmEpilogueDataType date_type = GemmEpilogueDataType::fp16;
    if constexpr(std::is_same<DataType_, __nv_bfloat16>::value){
        date_type = GemmEpilogueDataType::bf16;
    }
    else if constexpr(std::is_same<DataType_, float>::value){
        date_type = GemmEpilogueDataType::fp32;
    }
    else{
        ;
    }

    DataType_ *input, *weight, *bias, *output;
    CUDA_CHECK(cudaMalloc((void**)&input, sizeof(DataType_) * M * K));
    CUDA_CHECK(cudaMalloc((void**)&weight, sizeof(DataType_) * K * N));
    // 行主序的bias应该是N
    CUDA_CHECK(cudaMalloc((void**)&bias, sizeof(DataType_) *M* N));
    CUDA_CHECK(cudaMalloc((void**)&output, sizeof(DataType_) * M * N));
    nvtxRangeId_t initMatrix = nvtxRangeStartA("InitMatrix");
    InitMatrix(input, M, K);
    InitMatrix(weight, K, N);
    InitMatrix(bias, M, N);
    InitMatrix(output, M, N);
    nvtxRangeEnd(initMatrix);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // cutlass
    bool isVec_bias = false;
    void * workspace;
    size_t workspace_size_bytes = ((M-1+16)/16) * ((N-1+8)/8) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size_bytes));
    GemmEpilogueAllParams params{
        input, weight, bias, output, M, N, K, lda, ldb, ldd, stream, date_type, isVec_bias, sm_version, leaky_alpha, workspace
    };

    cutlass::Status status;
    float elapsed_time;
    for(auto func_name : all_func){
        /// warmup
        auto func = func_name.func;
        nvtxRangeId_t warmup_nvtx = nvtxRangeStartA("warmup_nvtx");
        
        status = func(params);

        nvtxRangeEnd(warmup_nvtx);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUTLASS_CHECK(status);

        /// run
        cudaEvent_t beg, end;
        CUDA_CHECK(cudaEventCreate(&beg));
        CUDA_CHECK(cudaEventCreate(&end));

        CUDA_CHECK(cudaEventRecord(beg));
        nvtxRangeId_t post_kernel_id = nvtxRangeStartA(func_name.funcName.c_str());
        status = func(params);
        nvtxRangeEnd(post_kernel_id);
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, beg, end));
        
        std::cout << func_name.funcName << " cost_time: " << elapsed_time << "ms." << std::endl;
        // float max_diff = gemm_epilogue_diff_gpu<DataType_>(params, op_type);
        // std::cout << max_diff << std::endl; 
    }
    
    /// HGEMM
    size_t C_Bytes = sizeof(DataType_) * M * N;
    DataType_ *h_C_cublas = (DataType_ *)malloc(C_Bytes);
    DataType_ *d_C_cublas;
    CUDA_CHECK(cudaMalloc((void**)&d_C_cublas, C_Bytes));
    callCublasHGEMM(M, N, K, input, weight, bias, h_C_cublas, C_Bytes);
    DataType_ *h_C_cutlass = (DataType_ *)malloc(C_Bytes);
    CUDA_CHECK(cudaMemcpy(h_C_cublas, d_C_cublas, C_Bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_cutlass, params.output, C_Bytes, cudaMemcpyDeviceToHost));
    validRes(h_C_cublas, h_C_cutlass, M*N);

    free(h_C_cublas);
    free(h_C_cutlass);
    CUDA_CHECK(cudaFree(d_C_cublas));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(weight));
    CUDA_CHECK(cudaFree(bias));
    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(workspace));
}

int main(){
    cudaSetDevice(3);
    /*
    int op_idx = 0;
    std::string activation = OpType2String(ops[op_idx]);
    if(activation == "matmul_add_leaky_relu")
        leaky_alpha = 0.01;
    for(int i = 0; i < 100; i+=7){
        int M = u(e);               // 
        int N = 8*(i+1);            // u(e);
        int K = 8*(i*3+7);
        std::cout << "MNK= [" << M << ", " <<  N << ", " << K << "], Act: " << activation << std::endl;
        EnumParamList_MNK_Act(M, N, K, ops[op_idx], activation, leaky_alpha);
    }
    */
    
    int M = 77;
    int N = 768;
    int K = 768;
    OpType op_type = OpType::MATMUL_ADD;
    std::string activation = OpType2String(op_type);
    EnumParamList_MNK_Act(M, N, K, op_type, activation, 1.f);
    
    return 0;
}

__global__ void InitMatrix_kernel(DataType_ *matrix, int rows, int cols){
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    if(i < rows && j < cols){
        int offset = i*cols + j;
        const int k = 16807;
        const int m = 16;
        float valToAssign = float((offset * k % m) - m/2);
        matrix[offset] = (DataType_)(valToAssign);
    }
}

void InitMatrix(DataType_ *matrix, int rows, int cols){
    dim3 block(16, 16);
    dim3 grid((cols+block.x-1)/block.x, (rows+block.y-1)/block.y);
    InitMatrix_kernel<<<grid, block>>>(matrix, rows, cols);
}

void validRes(DataType_ *h_C_cublas, DataType_ *h_C_cutlass, size_t C_size){
    float max_diff = 0.1;
    for(int i = 0; i < C_size; i++){
        float cutlass_value = static_cast<float>(h_C_cutlass[i]);
        float cublas_value = static_cast<float>(h_C_cublas[i]); 
        if (std::abs(cublas_value - cutlass_value) > max_diff) {
            float max_diff = std::abs(cublas_value - cutlass_value);
            std::cout << "diff: " << max_diff << std::endl;
            return;
        }
    }
    std::cout << "cublasRes == cutlassRes" << std::endl;
}


// for(auto func_name : all_func){
//     float total_time = 0.;
//     auto func = func_name.func;
//     nvtxRangeId_t warmup_nvtx = nvtxRangeStartA('warmup_nvtx');
//     for (int i = 0; i < 10; i++) {
//         status = func(params);
//     }
//     nvtxRangeEnd(warmup_nvtx);
//     CUDA_CHECK(cudaDeviceSynchronize());
//     cudaEvent_t beg, end;
//     CUDA_CHECK(cudaEventCreate(&beg));
//     CUDA_CHECK(cudaEventCreate(&end));
//     for(int i = 0; i < 10; i++){
//         CUDA_CHECK(cudaEventRecord(beg));
//         nvtxRangeId_t post_kernel_id = nvtxRangeStartA(func_name.funcName.c_str());
//         status = func(params);
//         nvtxRangeEnd(post_kernel_id);
//         CUDA_CHECK(cudaEventRecord(end));
//         CUDA_CHECK(cudaEventSynchronize(end));
//         CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, beg, end));
//         total_time += elapsed_time;
//         CUTLASS_CHECK(status);
//     }
//     std::cout << func_name.funcName << ' cost_time: ' << total_time << 'ms.' << std::endl;
// }

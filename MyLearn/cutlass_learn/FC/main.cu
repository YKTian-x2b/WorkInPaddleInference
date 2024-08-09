#include "paddle/phi/kernels/fusion/cutlass/fully_connected/fc_util.h"
#include "paddle/phi/kernels/fusion/cutlass/fully_connected/fc_bias_act_generate_example.cu"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

using DataType_ = half;

void InitMatrix(DataType_ *matrix, int rows, int cols);

int main(){
    int M = 1024, N = 1024, K = 2048;
    // 行主序
    int lda = K, ldb = N, ldc_bias = 0, ldd = N;
    OpType op_type = FC_BIAS_RELU;
    float alpha = 1.0;

    FcdDataType date_type = FcdDataType::fp16;
    if(std::is_same<DataType_, __nv_bfloat16>::value){
        date_type = FcdDataType::bf16;
    }
    else if(std::is_same<DataType_, float>::value){
        date_type = FcdDataType::fp32;
    }

    ///TODO:初始化和传输
    DataType_ *input, *weight, *bias, *output;
    CUDA_CHECK(cudaMalloc((void**)&input, sizeof(DataType_) * M * K));
    CUDA_CHECK(cudaMalloc((void**)&weight, sizeof(DataType_) * K * N));
    // 行主序的bias应该是N
    CUDA_CHECK(cudaMalloc((void**)&bias, sizeof(DataType_) * N));
    CUDA_CHECK(cudaMalloc((void**)&output, sizeof(DataType_) * M * N));
    InitMatrix(input, M, K);
    InitMatrix(input, K, N);
    InitMatrix(input, 1, N);
    InitMatrix(input, M, N);

    void *workspace;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    FcAllParams params{
        input, weight, bias, output, M, N, K, lda, ldb, ldd, alpha, stream, date_type, workspace
    };
    // cutlass
    fc_bias_relu_sm80_half_1(params);
    CUDA_CHECK(cudaDeviceSynchronize());
    // navie and diff
    float max_diff = fc_diff_gpu<DataType_>(params, op_type);
    std::cout << max_diff << std::endl;

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(weight));
    CUDA_CHECK(cudaFree(bias));
    CUDA_CHECK(cudaFree(output));

    return status;
}

__global__ void InitMatrix_kernel(DataType_ *matrix, int rows, int cols){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.x + blockIdx.y * blockDim.y;
    if(i < rows && j < cols){
        int offset = i + j * rows;
        const int k = 16807;
        const int m = 16;
        float valToAssign = float((offset * k % m) - m/2);
        if(std::is_same<DataType_, half>::value){
            matrix[offset] = __float2half(valToAssign);
        }
        else if(std::is_same<DataType_, __nv_bfloat16>::value){
            matrix[offset] = __float2bfloat16(valToAssign);
        }
        else{
            matrix[offset] = valToAssign;
        }
    }
}

void InitMatrix(DataType_ *matrix, int rows, int cols){
    dim3 block(16, 16);
    dim3 grid((rows+block.x-1)/block.x, (cols+block.y-1)/block.y);
    InitMatrix_kernel<<<grid, block>>>(matrix, rows, cols);
}
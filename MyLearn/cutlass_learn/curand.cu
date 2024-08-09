#include <curand_kernel.h>  
  
// 核函数，用于生成随机数  
__global__ void generateRandomNumbers(float* output, curandState* state, int N) {  
    int tid = threadIdx.x + blockIdx.x * blockDim.x;  
    if (tid < N) {  
        curandState localState = state[tid];  
        output[tid] = curand_uniform(&localState);  
        state[tid] = localState; // 更新状态（如果需要）  
    }  
}  
  
int main() {  
    // 初始化变量和内存  
    const int N = 1024; // 要生成的随机数的数量  
    float* h_output; // 主机上的输出数组  
    curandState* d_state; // 设备上的状态数组  
    float* d_output; // 设备上的输出数组  
  
    // 分配主机内存  
    h_output = (float*)malloc(N * sizeof(float));  
  
    // 分配设备内存  
    cudaMalloc((void**)&d_output, N * sizeof(float));  
    cudaMalloc((void**)&d_state, N * sizeof(curandState));  
  
    // 初始化随机数生成器的状态（在主机上）  
    curandState* h_state = (curandState*)malloc(N * sizeof(curandState));  
    for (int i = 0; i < N; ++i) {  
        curand_init(time(0), i, 0, &h_state[i]); // 使用当前时间和索引作为种子  
    }  
  
    // 将状态从主机复制到设备  
    cudaMemcpy(d_state, h_state, N * sizeof(curandState), cudaMemcpyHostToDevice);  
  
    // 调用核函数生成随机数  
    generateRandomNumbers<<<1, N>>>(d_output, d_state, N);  
  
    // 将输出从设备复制回主机  
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);  
  
    // 清理  
    free(h_output);  
    free(h_state);  
    cudaFree(d_output);  
    cudaFree(d_state);  
  
    return 0;  
}
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#define CHECK(call)                                                                  \
{                                                                                    \
    const cudaError_t error = call;                                                  \
    if(error != cudaSuccess){                                                        \
        printf("Error: %s: %d, ", __FILE__, __LINE__);                               \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));          \
        exit(1);                                                                     \
    }                                                                                \
}

__global__ void vecAdd(float * A, float * B, float * C){
    int idx = threadIdx.x;
    C[idx] = A[idx] + B[idx];
}

__host__ void warmup(){
    printf("warmup begin\n");
    size_t constexpr nBytes = 32 * sizeof(float);
    float * h_A = (float*)malloc(nBytes);
    float * h_B = (float*)malloc(nBytes);
    for(int i = 0; i < 32; ++i){
      h_A[i] = i;
      h_B[i] = i;
    }
    float * d_A, * d_B, * d_C;
    CHECK(cudaMalloc(&d_A, nBytes));
    CHECK(cudaMalloc(&d_B, nBytes));
    CHECK(cudaMalloc(&d_C, nBytes));
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    vecAdd<<<1,32>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    float * h_C = (float*)malloc(nBytes);
    CHECK(cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost));
    for(int i = 0; i < 32; ++i){
      printf("h_C[%d] = %f\n", i, h_C[i]);
    }

    printf("transpose before: %s\n", cudaGetErrorString(cudaGetLastError()));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    printf("warmup end\n");
}
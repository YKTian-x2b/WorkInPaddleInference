#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cstdio>

int main(){
    int count;
    cudaGetDeviceCount(&count);
    std::cout << "设备数量： " << count << std::endl;

    int device;
    // initDevice(5);
    cudaSetDevice(5);
    cudaGetDevice(&device);
    std::cout << "当前设备：" << device << std::endl;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "每个SM的寄存器总量：" << prop.regsPerMultiprocessor << std::endl;
    std::cout << "每个block的寄存器总量：" << prop.regsPerBlock << std::endl;
    std::cout << "总全局内存大小："  << prop.totalGlobalMem / 1024 / 1024 / 1024  << "GB" << std::endl;
    std::cout << "每个block的共享内存大小：" << prop.sharedMemPerBlock << std::endl;
    std::cout << "SM数量："  << prop.multiProcessorCount << std::endl;
    std::cout << "l2 cache大小：" << prop.l2CacheSize / 1024 / 1024 << "MB" << std::endl;
    std::cout << "每个SM最大线程数：" << prop.maxThreadsPerMultiProcessor << std::endl;
}

// nvcc A100_arch.cu -o res/A100_arch
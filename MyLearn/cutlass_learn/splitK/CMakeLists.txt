project (splitK)
cmake_minimum_required(VERSION 3.23)

set(fileList)
list(APPEND fileList "main.cu" "gemm_epilogue_util.cu" "gemm_epilogue_util.h" "gemm_epilogue_decl.h"
                     "matmul_add_sm80_fp16_15.cu" "matmul_add_sm80_fp16_81.cu" "matmul_add_sm80_fp16_99.cu")

message(status, ${fileList})
find_package(CUDA)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../../cutlass/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../../cutlass/tools/util/include")

set(CUDA_NVCC_FLAGS -std=c++17 -gencode arch=compute_80,code=sm_80 --ptxas-options --verbose;) 
CUDA_ADD_EXECUTABLE(splitK ${fileList})

target_link_libraries(splitK cublas)
target_link_libraries(splitK cudnn)
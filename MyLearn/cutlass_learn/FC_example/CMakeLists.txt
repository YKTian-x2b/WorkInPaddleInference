project (FC)
cmake_minimum_required(VERSION 3.1...3.21 FATAL_ERROR)

file(GLOB all_cu "*.cu")
find_package(CUDA)

#  git checkout v2.11.0
include_directories("/tyk/cutlass/include")
include_directories("/tyk/cutlass/tools/util/include")
include_directories("../")


# nvcc flags
set(CUDA_NVCC_FLAGS -gencode arch=compute_80,code=sm_80 -std=c++17 --ptxas-options --verbose;)
CUDA_ADD_EXECUTABLE(FC ${all_cu})
target_link_libraries(FC cublas)
target_link_libraries(FC cudnn)
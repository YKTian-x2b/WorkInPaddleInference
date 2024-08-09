set -e

# nvcc -lcublas -I/usr/local/cuda-11.6/include -arch=compute_80 -code=sm_80 hgemm_tensorcore.cu -o hgemm_tensorcore 

# ./hgemm_tensorcore


nvcc -lcublas -I/usr/local/cuda-11.6/include -arch=compute_80 -code=sm_80 hgemm_my.cu -o hgemm_my 
./hgemm_my
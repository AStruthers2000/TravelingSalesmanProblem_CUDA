#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void print_info()
{
    unsigned int thread_id = threadIdx.x;
    unsigned int block_id = blockIdx.x;
    unsigned int index = block_id * blockDim.x + thread_id;
    printf("Hello from thread %u on block %u. With a block dimension of %u, "
           "this thread is operating on index %u\n",
           thread_id, block_id, blockDim.x, index);
}

int main()
{
    print_info<<<4, 32>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}

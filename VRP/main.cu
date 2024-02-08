//system includes
#include <iostream>

//cuda includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//our includes
#include "Utils/HelperFunctions.cu"
#include "Problem/ProblemInstance.cu"

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

    //these are not allocated on the heap, so we don't have to free
    int test[4] = {1, 2, 3, 4};
    float test1[5] = {1.00012512, 2.3, 9.0, 0.2 + 0.1, 1e7};

    HelperFunctions::Host_PrintArray<int>(test, 4);
    HelperFunctions::Host_PrintArray<float>(test1, 5, 16);

    Node node_test[5] = {{40,50, 0}, {25, 85, 1}, {24, 29, 2}, {97, 2, 3}, {40, 50, 4}};
    HelperFunctions::Host_PrintArray<Node>(node_test, 5);


    return EXIT_SUCCESS;
}

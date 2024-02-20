//system includes
#include <iostream>

//cuda includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

//our includes
#include "Utils/HelperFunctions.cu"
#include "Problem/ProblemInstance.cu"

#define NUM_SM 80


__global__ void calculate_next_iteration(double** adj_mat, int* solutions, double* fitnesses, int solution_size);
void myexit();

__global__ void print_info()
{
    unsigned int thread_id = threadIdx.x;
    unsigned int block_id = blockIdx.x;
    unsigned int index = block_id * blockDim.x + thread_id;
    printf("Hello from thread %u on block %u. With a block dimension of %u, "
           "this thread is operating on index %u\n",
           thread_id, block_id, blockDim.x, index);
}

#define DEBUG false

int main()
{
    /*
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
     */

    int n_devices;
    cudaGetDeviceCount(&n_devices);
    printf("Number of CUDA devices: %d\n", n_devices);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties_v2(&prop, 0);
    printf("%s card information:\n", prop.name);
    printf("\tMax threads per block:    \t%d\n", prop.maxThreadsPerBlock);
    printf("\tMax threads per SM:       \t%d\n", prop.maxThreadsPerMultiProcessor);
    printf("\tMax thread blocks per SM: \t%d\n", prop.maxBlocksPerMultiProcessor);
    printf("\tMultiprocessor count:     \t%d\n", prop.multiProcessorCount);
    printf("\tMax grid size:            \t%d\n", prop.maxGridSize[0]);
    cudaSetDevice(0);

    static const int multiprocessors = prop.multiProcessorCount;
    static const int threads_per_block = prop.maxThreadsPerBlock;

    static const int test_numBlocks = 8;
    static const int test_threadsPerBlock = 32;

    int num_blocks = 0;
    int num_threads = 0;
    if(DEBUG)
    {
        num_blocks = test_numBlocks;
        num_threads = test_threadsPerBlock;
    }
    else
    {
        num_blocks = multiprocessors;
        num_threads = threads_per_block;
    }

    const int solution_size = 100;
    int total_length = num_blocks * num_threads * solution_size;
    int num_subsequences = total_length / solution_size;

    std::cout << "Generating initial solutions..." << std::endl;

    //create the repeating sequence 0, 1, 2, ..., solution_size - 1, 0, 1, 2, ...
    thrust::counting_iterator<int> solutions_begin(0);
    thrust::counting_iterator<int> solutions_end = solutions_begin + total_length;
    thrust::device_vector<int> solutions(total_length);
    thrust::transform(solutions_begin, solutions_end, solutions.begin(), [=]__device__(int x) { return x % solution_size; });

    // Create a permutation vector to shuffle each subsequence
    thrust::device_vector<int> permutation(solution_size);
    thrust::sequence(permutation.begin(), permutation.end());

    thrust::default_random_engine rng;
    // Shuffle each subsequence independently
    for (int i = 0; i < num_subsequences; ++i) {
        int offset = i * solution_size;
        thrust::device_vector<int>::iterator first = solutions.begin() + offset;
        thrust::device_vector<int>::iterator last = first + solution_size;
        thrust::shuffle(first, last, rng);
    }

    // Rearrange the elements of the sequence according to the shuffled permutation
    thrust::device_vector<int> temp(total_length);
    for (int i = 0; i < num_subsequences; ++i) {
        int offset = i * solution_size;
        thrust::device_vector<int>::iterator src_first = solutions.begin() + offset;
        thrust::device_vector<int>::iterator src_last = src_first + solution_size;
        thrust::device_vector<int>::iterator dst_first = temp.begin() + offset;
        thrust::gather(permutation.begin(), permutation.end(), src_first, dst_first);
    }
    solutions = temp;
    cudaDeviceSynchronize();
    int* device_solutions = thrust::raw_pointer_cast(solutions.data());

    std::cout << "Initial solutions generated" << std::endl;

    /*
    int* host_solutions;
    host_solutions = (int*) calloc(total_length, sizeof(int));
    cudaMemcpy(host_solutions, device_solutions, total_length * sizeof(int), cudaMemcpyDeviceToHost);



    std::cout << "here" << std::endl;
    for(int i = total_length - solution_size; i < total_length; i++)
    {
        if(i % solution_size == 0 && i > 0) std::cout << std::endl;
        std::cout << host_solutions[i] << ", ";

    }
    std::cout << std::endl;
     */

    calculate_next_iteration<<<num_blocks, num_threads>>>(nullptr, device_solutions, nullptr, solution_size);
    cudaDeviceSynchronize();


    atexit(myexit);
    return EXIT_SUCCESS;
}

__global__ void calculate_next_iteration(double** adj_mat, int* solutions, double* fitnesses, const int solution_size)
{
    auto thread_id = threadIdx.x;
    auto block_id = blockIdx.x;
    auto index = block_id * blockDim.x + thread_id;
    auto start = index * solution_size;
    auto end = start + solution_size;

    //if(thread_id == 0 || thread_id == 1023)
    if(index == 81919)
    {
        printf("Hello from thread %u on block %u aka index %u. I am operating on the solution solutions[%u, %u]. My fitness index is %u\n", thread_id, block_id, index, start, end - 1, index);
        //unsigned long encode = 0;
        for (auto i = start; i < end; ++i)
        {
            printf("%d ", solutions[i]);
            //encode += static_cast<unsigned long>(pow(10, (solution_size - i + start - 1))) * solutions[i];
            //printf("%d\n", (solution_size - i + start));
            //printf("%f\n", pow(10, (solution_size - i + start)));
        }

        printf("\n");
        //printf("%lu\n", encode);
    }
}

void myexit()
{
    printf("Exiting and resetting device\n");
    cudaError_t err = cudaDeviceReset();
    if(err != cudaSuccess)
    {
        printf("Error: %s", cudaGetErrorString(err));
    }
}
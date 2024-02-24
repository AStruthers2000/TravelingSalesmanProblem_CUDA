//system includes
#include <iostream>

//cuda includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//our includes
//#include "Utils/HelperFunctions.cu"
//#include "Problem/ProblemInstance.cu"
//#include "Algorithms/AntColony.cuh"
//#include "ModelParameters.h"
#include <iostream>
#include <math.h>

using namespace std;

// evaporation rate must be an element of [0, 1]
const double PHEROMONE_EVAPORATION_RATE = 0.01;



__global__ void copyMatrix(double* destinationMatrix, double* startingMatrix, int matrixWidth);

void myexit();

// A helper method for handling erros from CUDA calls
void cudaHandleError(cudaError_t error) {
    if (error != cudaSuccess) {
        cout << "Failed to perform device operation: " << cudaGetErrorString(error) << "\n";
        error = cudaDeviceReset();
    }
}
void getSTSPAdjacencyMatrix(double* matrix, string location, int problemSize) {}

void getATSPAdjacencyMatrix(double* matrix, string location, int nullKey) {}

void updatePheromoneMatrix(double* pheromoneMatrix, int pheromoneMatrixSize, double* antHistories, int antHistoriesSize, int numAnts, int iteration, int problemSize) {

    // perform pheromone evaporation on every edge
    for (int i = 0; i < problemSize; i++) {
        for (int j = 0; j < problemSize; j++) {
            pheromoneMatrix[i * problemSize + j] = pheromoneMatrix[i * problemSize + j] * PHEROMONE_EVAPORATION_RATE;
        }
    }



    // for each ant history
    for (int i = 0; i < numAnts; i++) {

        // for each edge traveled
        for (int j = 0; j <= problemSize; j++) {
            int startingCity = (int) antHistories[(problemSize + 1) * i + j];
            int endingCity;

            if (j == problemSize) { // loop back around to starting index if at the end of the path
                endingCity = (int) antHistories[(problemSize + 1) * i + 0];
            }
            else {
                endingCity = (int) antHistories[(problemSize + 1) * i + j + 1];
            }

            // at the edge traveled, update the pheromone matrix according to the fitness of the ant's solution

            // get the amount to add to the edge (ant's total tour length is stored at the last index of the history)
            double pheromoneToAdd = 1 / (antHistories[i * (problemSize + 1) + problemSize]);

            pheromoneMatrix[problemSize * startingCity + endingCity] = pheromoneMatrix[problemSize * startingCity + endingCity] + pheromoneToAdd;
        }
    }

}


void recordPheromoneMatrix(double* pheromone_matrix_per_iteration, double* pheromoneMatrix, int problemSize, int pheromoneMatrixSize, int iteration) {
    // allocate memory on the device for the pheromone matrix
    double* device_destinationPheromoneMatrix;
    cudaHandleError(cudaMalloc(&device_destinationPheromoneMatrix, pheromoneMatrixSize));

    double* device_startingPheromoneMatrix;
    cudaHandleError(cudaMalloc(&device_startingPheromoneMatrix, pheromoneMatrixSize));
    cudaHandleError(cudaMemcpy(device_startingPheromoneMatrix, pheromoneMatrix, pheromoneMatrixSize, cudaMemcpyHostToDevice));

    // get the number of threads and blocks.
    unsigned int num_threads = 128;
    unsigned int num_blocks = ceil(problemSize / 128);

    // kernel call
    copyMatrix<<<num_blocks, num_threads >>> (device_destinationPheromoneMatrix, device_startingPheromoneMatrix, problemSize);

    cudaHandleError(cudaMemcpy(&pheromone_matrix_per_iteration[iteration * problemSize * problemSize], device_destinationPheromoneMatrix, pheromoneMatrixSize, cudaMemcpyDeviceToHost));

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        cout << "Failed to perform device operation " << cudaGetErrorString(error);
        error = cudaDeviceReset();
    }

    // free device memory
    cudaHandleError(cudaFree(device_destinationPheromoneMatrix));
    cudaHandleError(cudaFree(device_startingPheromoneMatrix));
}

void recordBestAndAverageDistance(double* best_distance_per_iteration, double* average_distance_per_iteration, double* antHistories, int numAnts, int problemSize, int iteration) {
    double distanceSum = 0;

    // get the distance of the first ant
    double bestDistance = antHistories[0 * problemSize + problemSize];

    double curDistance;

    for (int i = 1; i < numAnts; i++) {
        // get the distance of the curent ant
        curDistance = antHistories[i * problemSize + problemSize];

        // add to the sum of distances
        distanceSum += curDistance;

        // if the curent distance is better than best distance, save it as the new best
        if (curDistance < bestDistance) {
            bestDistance = curDistance;
        }
    }

    // get the average distance
    double averageDistance = distanceSum / numAnts;

    // save best and average distance of the curent iteration.
    best_distance_per_iteration[iteration] = bestDistance;
    average_distance_per_iteration[iteration] = averageDistance;
}

void ACOsolve(double* adjacencyMatrix, int problemSize, int numAnts, int numIterations, double* best_distance_per_iteration, double* average_distance_per_iteration, double* pheromone_matrix_per_iteration) {

    
    // create coppies of the problem on the device
    int adjacencyMatrixSize = sizeof(double) * problemSize * problemSize;
    double* device_adjacencyMatrix;
    cudaHandleError(cudaMalloc(&device_adjacencyMatrix, adjacencyMatrixSize));
    cudaHandleError(cudaMemcpy(device_adjacencyMatrix, adjacencyMatrix, adjacencyMatrixSize, cudaMemcpyHostToDevice));
    


    // allocate pheromone matrix on host and device
    int pheromoneMatrixSize = sizeof(double) * problemSize * problemSize;
    double* pheromoneMatrix = (double*)malloc(pheromoneMatrixSize);

    double* device_pheromoneMatrix;
    cudaHandleError(cudaMalloc(&device_pheromoneMatrix, pheromoneMatrixSize));

    

    // allocate ant histories on matrix (with an additoinal comun to hold the distance traveled so far)
    int antHistoriesSize = sizeof(double) * numAnts * (problemSize + 1);
    double* antHistories = (double*)malloc(antHistoriesSize);

    double* device_antHistories;
    cudaHandleError(cudaMalloc(&device_antHistories, antHistoriesSize));

    
    // for the given number of iterations
    for (int i = 0; i < numIterations; i++) {
        // load pheromone matrix to device
        cout << "copy pheromone matrix to device\n";
        cudaHandleError(cudaMemcpy(device_pheromoneMatrix, pheromoneMatrix, pheromoneMatrixSize, cudaMemcpyHostToDevice));
        
        // invoke kernel
        // check for kernel errors (immediately after kernel execution)

        // retrieve ant histories
        cout << "copy histories from device to host\n";
        cudaHandleError(cudaMemcpy(antHistories, device_antHistories, antHistoriesSize, cudaMemcpyDeviceToHost));

        // update pheromone matrix
        cout << "update pheromone matrix\n";
        updatePheromoneMatrix(pheromoneMatrix, pheromoneMatrixSize, antHistories, antHistoriesSize, numAnts, i, problemSize);


        // record pheromone matrix
        cout << "record pheromone matrix\n";
        recordPheromoneMatrix(pheromone_matrix_per_iteration, pheromoneMatrix, problemSize, pheromoneMatrixSize, i);

        // record best distance and average distance
        cout << "record best and average tour\n";
        recordBestAndAverageDistance(best_distance_per_iteration, average_distance_per_iteration, antHistories, numAnts, problemSize, i);

        cout << "iteration: " << i << "\n";
    }




    // get ant histories and find best result




    // free all used memory

        // device
    cudaHandleError(cudaFree(device_pheromoneMatrix));
    cudaHandleError(cudaFree(device_antHistories));

    // host
    free(pheromoneMatrix);
    free(antHistories);
}


int main()
{

    // for a given problem size
    int STSPproblemSize = 1400; // number of cities
    int ATSPproblemSize = 65; // number of cites

    // and data in local file at a given location
    string STSPLocation = "fl1400.tsp";
    string ATSPLocation = "ftv64.atsp";

    // for a given number of ants
    int numAnts = 10000;

    // run a given number of iterations
    int numIterations = 100;

    // and possibly some null key for data integrity
    int nullKey = 100000000;

    // get STSP adjacencyMatrix
    double* STSP_adjacencyMatrix = (double*)malloc(sizeof(double) * STSPproblemSize * STSPproblemSize);
    getSTSPAdjacencyMatrix(STSP_adjacencyMatrix, STSPLocation, STSPproblemSize);

    // get ATSP adjacencyMatrix
    double* ATSP_adjacencyMatrix = (double*)malloc(sizeof(double) * ATSPproblemSize * ATSPproblemSize);
    getATSPAdjacencyMatrix(ATSP_adjacencyMatrix, ATSPLocation, nullKey);


    // CREATE MATRICIES TO HOLD DATA FROM PROGRAM RUNS
        // best tour distances for each iteration (1-Dimension array)
    double* STSP_best_tour_distances_for_each_iteration = (double*)malloc(sizeof(double) * numIterations);
    double* ATSP_best_tour_distances_for_each_iteration = (double*)malloc(sizeof(double) * numIterations);

    // average tour distance of each iteration (1-Dimension array)
    double* STSP_average_tour_distance_for_each_iteraiton = (double*)malloc(sizeof(double) * numIterations);
    double* ATSP_average_tour_diatance_for_each_iteration = (double*)malloc(sizeof(double) * numIterations);

    // pheromone matrix for each iteration (3-dimension array)
    double* STSP_pheromone_matrix_for_each_iteraiton = (double*)malloc(sizeof(double) * numIterations * STSPproblemSize * STSPproblemSize);
    double* ATSP_pheromone_matrix_for_each_iteration = (double*)malloc(sizeof(double) * numIterations * ATSPproblemSize * ATSPproblemSize);


    // solve problems
    ACOsolve(STSP_adjacencyMatrix, STSPproblemSize, numAnts, numIterations, STSP_best_tour_distances_for_each_iteration, STSP_average_tour_distance_for_each_iteraiton, STSP_pheromone_matrix_for_each_iteraiton);
    ACOsolve(ATSP_adjacencyMatrix, ATSPproblemSize, numAnts, numIterations, ATSP_best_tour_distances_for_each_iteration, ATSP_average_tour_diatance_for_each_iteration, ATSP_pheromone_matrix_for_each_iteration);


    // free adjacency matrices
    free(STSP_adjacencyMatrix);
    free(ATSP_adjacencyMatrix);


    // free saved information from each iteration
    free(STSP_best_tour_distances_for_each_iteration);
    free(ATSP_best_tour_distances_for_each_iteration);

    free(STSP_average_tour_distance_for_each_iteraiton);
    free(ATSP_average_tour_diatance_for_each_iteration);

    free(STSP_pheromone_matrix_for_each_iteraiton);
    free(ATSP_pheromone_matrix_for_each_iteration);

    // ================== KERNEL CALLS =========================
    //ACOPrint << <GROUPS_OF_N_ANTS, THREADS_PER_BLOCK >> > ();
    //cudaDeviceSynchronize();
    //atexit(myexit);
    //return EXIT_SUCCESS;
}


__global__ void copyMatrix(double* destinationMatrix, double* startingMatrix, int matrixWidth) {
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;

    unsigned int index = block_id * thread_id;

    // while the index wont go out of bounds of the matrices
    if (index < matrixWidth) {

        // use the kernel to copy the index's row
        for (int i = 0; i < matrixWidth; i++) {

            destinationMatrix[matrixWidth * index + i] = startingMatrix[matrixWidth * index + i];
        }
    }
}

//
//__global__ void calculate_next_iteration(double** adj_mat, int* solutions, double* fitnesses, int solution_size);

//
//__global__ void print_info()
//{
//    unsigned int thread_id = threadIdx.x;
//    unsigned int block_id = blockIdx.x;
//    unsigned int index = block_id * blockDim.x + thread_id;
//    printf("Hello from thread %u on block %u. With a block dimension of %u, "
//           "this thread is operating on index %u\n",
//           thread_id, block_id, blockDim.x, index);
//}
//
//#define DEBUG false
//
//int main()
//{
//    /*
//    print_info<<<4, 32>>>();
//    cudaDeviceSynchronize();
//    cudaDeviceReset();
//
//    //these are not allocated on the heap, so we don't have to free
//    int test[4] = {1, 2, 3, 4};
//    float test1[5] = {1.00012512, 2.3, 9.0, 0.2 + 0.1, 1e7};
//
//    HelperFunctions::Host_PrintArray<int>(test, 4);
//    HelperFunctions::Host_PrintArray<float>(test1, 5, 16);
//
//    Node node_test[5] = {{40,50, 0}, {25, 85, 1}, {24, 29, 2}, {97, 2, 3}, {40, 50, 4}};
//    HelperFunctions::Host_PrintArray<Node>(node_test, 5);
//     */
//
//    int n_devices;
//    cudaGetDeviceCount(&n_devices);
//    printf("Number of CUDA devices: %d\n", n_devices);
//    cudaDeviceProp prop{};
//    cudaGetDeviceProperties_v2(&prop, 0);
//    printf("%s card information:\n", prop.name);
//    printf("\tMax threads per block:    \t%d\n", prop.maxThreadsPerBlock);
//    printf("\tMax threads per SM:       \t%d\n", prop.maxThreadsPerMultiProcessor);
//    printf("\tMax thread blocks per SM: \t%d\n", prop.maxBlocksPerMultiProcessor);
//    printf("\tMultiprocessor count:     \t%d\n", prop.multiProcessorCount);
//    printf("\tMax grid size:            \t%d\n", prop.maxGridSize[0]);
//    cudaSetDevice(0);
//
//    static const int multiprocessors = prop.multiProcessorCount;
//    static const int threads_per_block = prop.maxThreadsPerBlock;
//
//    static const int test_numBlocks = 8;
//    static const int test_threadsPerBlock = 32;
//
//    int num_blocks = 0;
//    int num_threads = 0;
//    if(DEBUG)
//    {
//        num_blocks = test_numBlocks;
//        num_threads = test_threadsPerBlock;
//    }
//    else
//    {
//        num_blocks = multiprocessors;
//        num_threads = threads_per_block;
//    }
//
//    const int solution_size = 100;
//    int total_length = num_blocks * num_threads * solution_size;
//    int num_subsequences = total_length / solution_size;
//
//    std::cout << "Generating initial solutions..." << std::endl;
//
//    //create the repeating sequence 0, 1, 2, ..., solution_size - 1, 0, 1, 2, ...
//    thrust::counting_iterator<int> solutions_begin(0);
//    thrust::counting_iterator<int> solutions_end = solutions_begin + total_length;
//    thrust::device_vector<int> solutions(total_length);
//    thrust::transform(solutions_begin, solutions_end, solutions.begin(), [=]__device__(int x) { return x % solution_size; });
//
//    // Create a permutation vector to shuffle each subsequence
//    thrust::device_vector<int> permutation(solution_size);
//    thrust::sequence(permutation.begin(), permutation.end());
//
//    thrust::default_random_engine rng;
//    // Shuffle each subsequence independently
//    for (int i = 0; i < num_subsequences; ++i) {
//        int offset = i * solution_size;
//        thrust::device_vector<int>::iterator first = solutions.begin() + offset;
//        thrust::device_vector<int>::iterator last = first + solution_size;
//        thrust::shuffle(first, last, rng);
//    }
//
//    // Rearrange the elements of the sequence according to the shuffled permutation
//    thrust::device_vector<int> temp(total_length);
//    for (int i = 0; i < num_subsequences; ++i) {
//        int offset = i * solution_size;
//        thrust::device_vector<int>::iterator src_first = solutions.begin() + offset;
//        thrust::device_vector<int>::iterator src_last = src_first + solution_size;
//        thrust::device_vector<int>::iterator dst_first = temp.begin() + offset;
//        thrust::gather(permutation.begin(), permutation.end(), src_first, dst_first);
//    }
//    solutions = temp;
//    cudaDeviceSynchronize();
//    int* device_solutions = thrust::raw_pointer_cast(solutions.data());
//
//    std::cout << "Initial solutions generated" << std::endl;
//
//    /*
//    int* host_solutions;
//    host_solutions = (int*) calloc(total_length, sizeof(int));
//    cudaMemcpy(host_solutions, device_solutions, total_length * sizeof(int), cudaMemcpyDeviceToHost);
//
//
//
//    std::cout << "here" << std::endl;
//    for(int i = total_length - solution_size; i < total_length; i++)
//    {
//        if(i % solution_size == 0 && i > 0) std::cout << std::endl;
//        std::cout << host_solutions[i] << ", ";
//
//    }
//    std::cout << std::endl;
//     */
//
//    calculate_next_iteration<<<num_blocks, num_threads>>>(nullptr, device_solutions, nullptr, solution_size);
//    cudaDeviceSynchronize();
//
//
//    atexit(myexit);
//    return EXIT_SUCCESS;
//}
//
//__global__ void calculate_next_iteration(double** adj_mat, int* solutions, double* fitnesses, const int solution_size)
//{
//    auto thread_id = threadIdx.x;
//    auto block_id = blockIdx.x;
//    auto index = block_id * blockDim.x + thread_id;
//    auto start = index * solution_size;
//    auto end = start + solution_size;
//
//    //if(thread_id == 0 || thread_id == 1023)
//    if(index == 81919)
//    {
//        printf("Hello from thread %u on block %u aka index %u. I am operating on the solution solutions[%u, %u]. My fitness index is %u\n", thread_id, block_id, index, start, end - 1, index);
//        //unsigned long encode = 0;
//        for (auto i = start; i < end; ++i)
//        {
//            printf("%d ", solutions[i]);
//            //encode += static_cast<unsigned long>(pow(10, (solution_size - i + start - 1))) * solutions[i];
//            //printf("%d\n", (solution_size - i + start));
//            //printf("%f\n", pow(10, (solution_size - i + start)));
//        }
//
//        printf("\n");
//        //printf("%lu\n", encode);
//    }
//}
//
void myexit()
{
    printf("Exiting and resetting device\n");
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        printf("Error: %s", cudaGetErrorString(err));
    }
}
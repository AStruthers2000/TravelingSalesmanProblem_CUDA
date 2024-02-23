
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


#include <iostream>
#include <string>
#include <string.h>
using namespace std;


void getSTSPAdjacencyMatrix(double* matrix, string location, int problemSize) {}

void getATSPAdjacencyMatrix(int* matrix, string location, int nullKey) {}


void ACOsolveSTSP(int problemSize, string location, int numAnts, int numIterations){
    // populate an adjacency matrix of the problem
    double* adjacencyMatrix = (double*)malloc(sizeof(double) * STSPproblemSize * STSPproblemSize);


    getSTSPAdjacencyMatrix(adjacencyMatrix, location, problemSize);


    // create coppies of the problems on the device
    double* device_adjacencyMatrix;
    cudaHandleError(cudaMalloc(&device_adjacencyMatrix, sizeof(double) * problemSize * problemSize));
    cudaHandleError(cudaMemcpy(device_adjacencyMatrix, adjacencyMatrix, sizeof(double) * problemSize * problemSize, cudaMemcpyHostToDevice));



    // allocate pheromone matrix on host and device
    double* pheromoneMatrix = (double*)malloc(sizeof(double) * problemSize * problemSize);

    double* device_pheromoneMatrix;
    cudaHandleError(cudaMalloc(&device_pheromoneMatrix, sizeof(double) * problemSize * problemSize));



    // allocate ant histories on matrix
    int* antHistories = (int*)malloc(sizeof(int) * numAnts * problemSize);

    int* device_antHistories;
    cudaHandleError(cudaMalloc(&device_antHistories, sizeof(int) * numAnts * problemSize));

    // invoke kernel

    // check for kernel errors (immediately after kernel execution)



    // get ant histories and find best result




    // free all used memory

        // device
    cudaHandleError(cudaFree(device_adjacencyMatrix));
    cudaHandleError(cudaFree(device_pheromoneMatrix));
    cudaHandleError(cudaFree(device_antHistories));

        // host
    free(adjacencyMatrix);
    free(pheromoneMatrix);
    free(antHistories);
}

void ACOsolveATSP(int problemSIze, string location, int numAnts, int numIterations, int nullKey){
    // populate an adjacency matrix of the problem
    int* adjacencyMatrix = (int*)malloc(sizeof(int) * problemSize * problemSize);

    getATSPAdjacencyMatrix(adjacencyMatrix, location, nullKey);


    // create coppies of the problems on the device
    int* device_adjacencyMatrix;
    cudaHandleError(cudaMalloc(&device_adjacencyMatrix, sizeof(int) * problemSize * problemSize));
    cudaHandleError(cudaMemcpy(device_adjacencyMatrix, adjacencyMatrix, sizeof(int) * problemSize * problemSize, cudaMemcpyHostToDevice));


    // allocate pheromone matrix on host and device
    double* pheromoneMatrix = (double*)malloc(sizeof(double) * problemSize * problemSize);
    
    double* device_pheromoneMatrix;
    cudaHandleError(cudaMalloc(&device_pheromoneMatrix, sizeof(double) * problemSize * problemSize));


    // allocate ant histories on matrix
    int* antHistories = (int*)malloc(sizeof(int) * numAnts * problemSize);

    int* device_antHistories;
    cudaHandleError(cudaMalloc(&device_antHistories, sizeof(int) * numAnts * problemSize));

    // invoke kernel

    // check for kernel errors (immediately after kernel execution)



    // get ant histories and find best result
    



    // free all used memory

        // device
    cudaHandleError(cudaFree(device_ATSPAdjacencyMatrix));
    cudaHandleError(cudaFree(device_ATSPpheromoneMatrix));
    cudaHandleError(cudaFree(device_ATSPAntHistories));

        // host
    free(ATSPAdjacencyMatrix);
    free(ATSPpheromoneMatrix);
    free(ATSPAntHistories);

}

void cudaHandleError(cudaError_t error) {
    if (error != cudaSuccess) {
        cout << "Failed to perform device operation: " << cudaGetErrorString(error);
    }
}

int main() {


    // for a given problem size
    int STSPproblemSize = 1400; // number of cities
    int ATSPproblemSize = 65; // number of cites

    // and data in local file at a given location
    string STSPLocation = "fl1400.tsp";
    string ATSPLocation = "ftv64.atsp";

    // for a given number of ants
    int numAnts = 10000;

    // run a given number of iterations
    int numIterations = 1000;

    // and possibly some null key for data integrity
    int nullKey = 100000000;

    ACOsolveSTSP(STSPproblemSize, STSPLocation, numAnts, numIterations);
    ACOsolveATSP(ATSPproblemSize, ATSPLocation, numAnts, numIterations, nullKey);

}
//system includes
#include <iostream>

//cuda includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>


#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>

using namespace std;


// set up constants
const int THREADS_PER_BLOCK = 128;


// define device constants

// evaporation rate must be an element of [0, 1]
__constant__ double PHEROMONE_EVAPORATION_RATE = 0.6;

// constants to regulate the influence of peheromones and edge weights
__constant__ double ALPHA = 4; // pheromone regulation
__constant__ double BETA = 6; // edge weight regulation

__constant__ double Q = 1;

// define device functions
__global__ void move_ant(double* matrix_pheromones, double* matrix_distances, curandState* states, double* history_distances, int* history_tours, int* history_visited, int num_cities, int numAnts, int curIteration, int threadsPerBlock);
__global__ void setup_curand_states(curandState* dev_states, unsigned long seed, int numAnts, int threadsPerBlock);
__global__ void pheromoneAdjust(double* pheromoneMatrix, int pheromoneMatrixSize, int* antHistories, double* antDistanceHistories, int antHistoriesSize, int numAnts, int threadsPerBlock);
__global__ void pheromoneMatrixEvaporation(double* pheromoneMatrix, int problemSzie, int threadsPerBlock);
__global__ void distanceReset(double* distanceHistory, int numAnts, int problemSize, int threadsPerBlock);

__device__ unsigned int getIndex(int threadsPerBlock);

void myexit();

// A helper method for handling erros from CUDA calls
void cudaHandleError(cudaError_t error) {
    if (error != cudaSuccess) {
        cout << "Failed to perform device operation: " << cudaGetErrorString(error) << "\n";
        error = cudaDeviceReset();
    }
}

void checkKernelError() {
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        cout << "Failed to perform device operation " << cudaGetErrorString(error) << "\n";
        error = cudaDeviceReset();
    }
}
void getSTSPAdjacencyMatrix(double* matrix, string location, int problemSize) {
    // create and open file stream
    ifstream tsp;
    tsp.open(location, ios::in);

    // create value to hold the line currently being read.
    string line;

    // create an integer i to indicate the value of the node being read.
    int i = 0;

    // create an array and allocate memory to hold the location of each point
    double* pointsLocations = (double*)malloc(sizeof(double) * problemSize * 2);

    // for every line in the .tsp data file.
    while (getline(tsp, line)) {

        // if the line is the end of the file, end the loop
        if (line == "EOF") break;

        // create a string stream to read the values in the line.
        stringstream ssm(line);

        // create and read values on each line.
        int index = 0;
        double x = 0;
        double y = 0;

        if (ssm >> index >> x >> y) {}
        else { continue; } // if there is an error reading the line, continue to the next iteration.

        // store the x and y coordinates in the pointsLocations matrix
        pointsLocations[i] = x;
        pointsLocations[i + 1] = y;

        // incriment i by 2.
        i += 2;
    }

    // using pointsLocations, create an adjacency matrix by euclidian distance

    // for every point
    for (i = 0; i < problemSize * 2; i += 2) {

        // get the x and y values of the curent point.
        double origin_x = 0;
        double origin_y = 0;

        origin_x = pointsLocations[i];
        origin_y = pointsLocations[i + 1];

        // for every other point
        for (int j = 0; j < problemSize * 2; j += 2) {

            // get the coordinates of the other point
            double second_x = 0;
            double second_y = 0;

            second_x = pointsLocations[j];
            second_y = pointsLocations[j + 1];

            // calculate the euclidean distance between the points and save it in the adjacency matrix.
            matrix[(i / 2) * problemSize + (j / 2)] = sqrt(pow(abs(second_x - origin_x), 2) + pow(abs(second_y - origin_y), 2));
        }
    }

    // close point and free memory
    tsp.close();
    free(pointsLocations);

}

void getATSPAdjacencyMatrix(double* matrix, string location, int nullKey) {

    // create and open the file stream to read ATSP data
    ifstream tsp;
    tsp.open(location, ios::in);

    // line to hold the curent line being read
    string line;

    // the curent index of data being accessed and assigned in the matrix and file.
    int i = 0;

    // for every line in the document
    while (getline(tsp, line)) {

        // create a string stream to read the values in the curent line.
        stringstream ssm(line);

        // create a temporary int to hold the value currently being accessed in the file.
        int cur = 0;

        // for every integer in the line
        while (ssm >> cur) {
            if (cur == nullKey) { // the the curent node is being evaluated save 0 at i
                matrix[i] = 0;
            }
            else {   // otherwise save the value
                matrix[i] = cur;
            }

            i++;
        }
    }
}

void printArray(double* arrayToPrint, int size) {
    for (int i = 0; i < size; i++) {
        cout << arrayToPrint[i] << ", ";
    }

    cout << "\n";
}
void printArray(int* arrayToPrint, int size) {
    for (int i = 0; i < size; i++) {
        cout << arrayToPrint[i] << ", ";
    }

    cout << "\n";
}


void printMatrix(double* matrixToPrint, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cout << matrixToPrint[width * i + j] << ", ";
        }
        cout << "\n";
    }
}

// updates pheormone matrix on device
void updatePheromoneMatrix(double* device_pheromoneMatrix, int pheromoneMatrixSize, int* device_antHistories, double* device_distanceHistories, int antHistoriesSize, int numAnts, int iteration, int problemSize) {

    // get the number of threads and blocks for the pheromone matrix
    int num_threads = 128;
    int num_blocks = ceil((double)problemSize / num_threads);

    // kernel call
    pheromoneMatrixEvaporation << <num_blocks, num_threads >> > (device_pheromoneMatrix, problemSize, THREADS_PER_BLOCK);
    // check for device error
    checkKernelError();

    cudaDeviceSynchronize();

    // update pheromone matrix based on ant histories
    num_blocks = ceil((double)numAnts / num_threads);

    pheromoneAdjust << <num_blocks, num_threads >> > (device_pheromoneMatrix, pheromoneMatrixSize, device_antHistories, device_distanceHistories, problemSize, numAnts, THREADS_PER_BLOCK);

    checkKernelError();

    cudaDeviceSynchronize();

}


void recordPheromoneMatrix(double* pheromone_matrix_per_iteration, double* device_pheromoneMatrix, int problemSize, int pheromoneMatrixSize, int iteration) {

    cudaHandleError(cudaMemcpy(&pheromone_matrix_per_iteration[iteration * pheromoneMatrixSize / sizeof(double)], device_pheromoneMatrix, pheromoneMatrixSize, cudaMemcpyDeviceToHost));
}

void recordBestAndAverageDistance(double* best_distance_per_iteration, double* average_distance_per_iteration, double* device_antDistances, int* device_antHistory, int numAnts, int problemSize, int iteration) {
    
    // copy distance history from device to host
    double* distanceHistory = (double*) malloc(sizeof(double) * numAnts);
    cudaHandleError(cudaMemcpy(distanceHistory, device_antDistances, sizeof(double) * numAnts, cudaMemcpyDeviceToHost));
    
    int* antHistory = (int*)malloc(sizeof(int) * numAnts * problemSize);
    cudaHandleError(cudaMemcpy(antHistory, device_antHistory, sizeof(int) * numAnts * problemSize, cudaMemcpyDeviceToHost));

    // set the best to the first index
    double best = distanceHistory[0];
    
    // initalize the sum of all distanes
    double sumOfDistances = distanceHistory[0];
    int bestAntIndex = 0;
    for (int i = 1; i < numAnts; i++) {
        //cout << i << ", \n";
        double curDist = distanceHistory[i];

        if (curDist < best) {
            //cout << "BETTER FOUND\n";
            bestAntIndex = i;
            best = curDist;
        }

        sumOfDistances += curDist;
        
    }

    

    // print path of best route
    /*
    for (int i = 0; i < problemSize; i++) {
        cout << antHistory[bestAntIndex * problemSize + i] << ", ";
    }

    cout << "\n";
    */
    best_distance_per_iteration[iteration] = best;
    average_distance_per_iteration[iteration] = sumOfDistances / numAnts;
    

    free(distanceHistory);
    free(antHistory);
}

void pheromoneInit(double* pheormoneMatrix, int problemSize) {
    for (int i = 0; i < problemSize; i++) {
        for (int j = 0; j < problemSize; j++) {
            pheormoneMatrix[i * problemSize + j] = 1;
        }
    }
}

void distanceInit(double* device_distanceHistory, int numAnts, int problemSize) {

    distanceReset << <1, 1 >> > (device_distanceHistory, numAnts, problemSize, THREADS_PER_BLOCK);

    checkKernelError();

    cudaDeviceSynchronize();
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

    pheromoneInit(pheromoneMatrix, problemSize);
    

    double* device_pheromoneMatrix;
    cudaHandleError(cudaMalloc(&device_pheromoneMatrix, pheromoneMatrixSize));



    // allocate ant histories on matrix (with an additoinal comun to hold the distance traveled so far)
    int antHistoriesSize = sizeof(int) * numAnts * (problemSize);
    int* antHistories = (int*)malloc(antHistoriesSize);

    int* device_antHistories;
    cudaHandleError(cudaMalloc(&device_antHistories, antHistoriesSize));

    // initialize curand state
    curandState* device_curandStates;
    cudaHandleError(cudaMalloc(&device_curandStates, sizeof(curandState) * numAnts));

    // initialize ant distance history
    double* device_distanceHistory;
    cudaHandleError(cudaMalloc(&device_distanceHistory, sizeof(double) * numAnts));

    // initialize visited cities
    int* device_visitedCities;
    cudaHandleError(cudaMalloc(&device_visitedCities, sizeof(int) * numAnts * problemSize));

    // get seed
    time_t seed;
    time(&seed);
    int num_blocks = ceil((double)numAnts / THREADS_PER_BLOCK);
    

    // initialize curand states on device
    setup_curand_states << <num_blocks, THREADS_PER_BLOCK >> > (device_curandStates, (unsigned long)seed, numAnts, THREADS_PER_BLOCK);

    checkKernelError();

    cudaDeviceSynchronize();

    // for the given number of iterations
    for (int i = 0; i < numIterations; i++) {
        // reset/initialize distanceHistory
        distanceInit(device_distanceHistory, numAnts, problemSize);

        // load pheromone matrix to device
        cout << "iteration " << i << "\n";
        cudaHandleError(cudaMemcpy(device_pheromoneMatrix, pheromoneMatrix, pheromoneMatrixSize, cudaMemcpyHostToDevice));

        //printMatrix(pheromoneMatrix, problemSize, problemSize);
        // invoke kernel
        int num_blocks = ceil((double)numAnts / THREADS_PER_BLOCK);

        move_ant << <num_blocks, THREADS_PER_BLOCK >> > (device_pheromoneMatrix, device_adjacencyMatrix, device_curandStates, device_distanceHistory, device_antHistories, device_visitedCities, problemSize, numAnts, i, THREADS_PER_BLOCK);
        // check for kernel errors (immediately after kernel execution)
        cudaDeviceSynchronize();
        checkKernelError();


        // retrieve ant histories
        cudaHandleError(cudaMemcpy(antHistories, device_antHistories, antHistoriesSize, cudaMemcpyDeviceToHost));

        // update pheromone matrix
        updatePheromoneMatrix(device_pheromoneMatrix, pheromoneMatrixSize, device_antHistories, device_distanceHistory, antHistoriesSize, numAnts, i, problemSize);

        cudaHandleError(cudaMemcpy(pheromoneMatrix, device_pheromoneMatrix, pheromoneMatrixSize, cudaMemcpyDeviceToHost));
        // record pheromone matrix
        recordPheromoneMatrix(pheromone_matrix_per_iteration, device_pheromoneMatrix, problemSize, pheromoneMatrixSize, i);

        // record best distance and average distance
        recordBestAndAverageDistance(best_distance_per_iteration, average_distance_per_iteration, device_distanceHistory, device_antHistories, numAnts, problemSize, i);
    }




    // get ant histories and find best result




    // free all used memory

        // device
    cudaHandleError(cudaFree(device_pheromoneMatrix));
    cudaHandleError(cudaFree(device_antHistories));
    cudaHandleError(cudaFree(device_curandStates));
    cudaHandleError(cudaFree(device_distanceHistory));
    cudaHandleError(cudaFree(device_visitedCities));

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
    //ACOsolve(STSP_adjacencyMatrix, STSPproblemSize, numAnts, numIterations, STSP_best_tour_distances_for_each_iteration, STSP_average_tour_distance_for_each_iteraiton, STSP_pheromone_matrix_for_each_iteraiton);
    ACOsolve(ATSP_adjacencyMatrix, ATSPproblemSize, numAnts, numIterations, ATSP_best_tour_distances_for_each_iteration, ATSP_average_tour_diatance_for_each_iteration, ATSP_pheromone_matrix_for_each_iteration);

    printArray(ATSP_best_tour_distances_for_each_iteration, numIterations);

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





__global__ void move_ant(double* matrix_pheromones, double* matrix_distances, curandState* states, double* history_distances, int* history_tours, int* history_visited, int num_cities, int numAnts, int curIteration, int threadsPerBlock) {
    //where the ant's history starts
    unsigned int ant_id = getIndex(threadsPerBlock);

    if (ant_id < numAnts) {

        //we will get an even distribution of ants starting at the various cities
        //with roughly an even number of ants starting at each city
        int current_city = (ant_id) % num_cities;
        history_visited[ant_id * num_cities + current_city] = ((curIteration + 1) % 2);
        history_tours[ant_id * num_cities] = current_city;


        //printf("Thread %d:%d starts at city %d\n", blockIdx.x, threadIdx.x, current_city);

        //we will iterate n - 1 times, because we already know where we want to start
        //on each iteration, we will calculate the probabilities of visiting each city
        //that this ant hasn't visited yet, then use roulette wheel selection to visit
        //a city determined probabilistically. Once this loop completes, we need to also
        //add the distance travelled from the last visited city, back to the beginning
        //because a tour is only finished once we return to where we started
        for (int i = 1; i < num_cities; i++)
        {

            //printf("new i %d", i);
            //we only care about accumulating probability for cities we have not visited
            //we want to compute:
            //\sum_{m\in allowed} \tau_{im}^\alpha \cdot \eta_{im}^\beta
            //where allowed is all cities we haven't visited yet
            double total_prob = 0.0;

            //printf("movement step %d\n", i);
            for (int next_city = 0; next_city < num_cities; next_city++)
            {
                
                //aka we have not yet visited this city
                //printf("city num %d\n", next_city);
                if (history_visited[ant_id * num_cities + next_city] != (curIteration + 1) % 2)
                {
                    //printf("%d ", next_city);
                    int city_index = current_city * num_cities + next_city;
                    double tau = matrix_pheromones[city_index];
                    double eta = 1/matrix_distances[city_index];
                    total_prob += pow(tau, ALPHA) * pow(eta, BETA);
                    
                    //printf("tau = %f || eta = %f || total_prob = %f || etc = %f\n", tau, eta, total_prob, pow(tau, ALPHA) * pow(eta, BETA));
                }
            }
            //printf("\n");
            //printf("end movement step %d total_prob = %f \n", i, total_prob);

            //perform roulette wheel selection to select our next city

            double r = curand_uniform(&states[ant_id]);
            double accum_prob = 0.0;
            int selected_city = -1;

            //printf("Thread %d:%d generated %2.2f\n", blockIdx.x, threadIdx.x, r);
            //for each possible city, we want to compute:
            //\tau_{im}^\alpha \cdot \eta_{im}^\beta
            //so that we can calculate P_{ij} as this product divided by the sum calculated above
            int last_best_city;
            for (int next_city = 0; next_city < num_cities; next_city++)
            {
                if (history_visited[ant_id * num_cities + next_city] != (curIteration + 1) % 2)
                {
                    last_best_city = next_city;

                    int city_index = current_city * num_cities + next_city;
                    
                    double tau = matrix_pheromones[city_index];
                    double eta = 1/matrix_distances[city_index];

                   
                    double num = pow(tau, ALPHA) * pow(eta, BETA);
                    
                    double prob = num / total_prob;

                    accum_prob += prob;
                    //printf("city %d || tau %f || eta %f || num %f || r %f || accumProb %f || totalProb %f\n", next_city, tau, eta, num, r, accum_prob, total_prob);
                    //printf("thread %d: %f >= %f\n", ant_id, accum_prob, r);
                    //printf("\tThread %d ?-> %d, with a probability of %2.2f where accum prob = %2.2f\n",threadIdx.x, next_city, prob, accum_prob);

                    if (accum_prob >= r)
                    {
                        //printf("thread %d new\n", ant_id);
                        selected_city = next_city;
                        break;
                    }
                }
            }

            //printf("Thread %d || city %d\n", ant_id, selected_city);

            //now that we have selected a city using the probability equation
            //we want to actually move to that city.
            //We need to add this city to this ant's tour, add this city to this
            //ant's visited history, and add the distance from this move to this ant's
            //distance history

            if (selected_city < 0) {
                selected_city = last_best_city;
                printf("ERROR -1 || %d || accumProb %f || r %f\n", ant_id, accum_prob, r);
            }
            
            //printf("history tours\n");
            history_tours[ant_id * num_cities + i] = selected_city;
            //printf("history visited\n");
            history_visited[ant_id * num_cities + selected_city] = ((curIteration + 1) % 2);
            //printf("history distances\n");
            history_distances[ant_id] += matrix_distances[current_city * num_cities + selected_city];

            current_city = selected_city;
        }
        //we need to return to the starting city now
        //we go from the last visited city
        int from = history_tours[ant_id * num_cities + (num_cities - 1)];
        //to the first visited city
        int to = history_tours[ant_id * num_cities];
        history_distances[ant_id] += matrix_distances[from * num_cities + to];
        
    }
}



__global__ void setup_curand_states(curandState* dev_states, unsigned long seed, int numAnts, int threadsPerBlock)
{
    unsigned int index = getIndex(threadsPerBlock);
    if (index < numAnts) {
        curand_init(seed, index, 0, &dev_states[index]);
    }
}

__device__ unsigned int getIndex(int numThreadsPerBlock) {
    return threadIdx.x + blockIdx.x*numThreadsPerBlock;
}

__global__ void pheromoneAdjust(double* pheromoneMatrix, int pheromoneMatrixSize, int* antHistories, double* antDistanceHistories, int problemSize, int numAnts, int threadsPerBlock) {


    unsigned int ant_index = getIndex(threadsPerBlock);


    if (ant_index < numAnts) {

        // for each edge traveled
        for (int i = 0; i < problemSize; i++) {
            // get the edge traveled
            int startingCity = (int)antHistories[problemSize * ant_index + i];
            int endingCity;

            if (i == problemSize-1) { // if at the last index, the ending city will be the starting city
                endingCity = (int)antHistories[problemSize * ant_index];
            }
            else {
                endingCity = (int)antHistories[problemSize * ant_index + i + 1];
            }
            // at the edge traveled, update the pheromone matrix according to the fitness of the ant's solution
            // get the amount to add to the edge (ant's total tour length is stored at the last index of the history)
            double pheromoneToAdd = Q/antDistanceHistories[ant_index];

            //printf("andID %d || pheromone to add %f || distance of ant %f\n", ant_index, pheromoneToAdd, antDistanceHistories[ant_index]);

            //printf("Pheromone to add %f\n", pheromoneToAdd);

            pheromoneMatrix[problemSize * startingCity + endingCity] = pheromoneMatrix[problemSize * startingCity + endingCity] + pheromoneToAdd;

        }
    }

}

__global__ void pheromoneMatrixEvaporation(double* pheromoneMatrix, int problemSize, int threadsPerBlock) {
    // get the curent thread's index

    unsigned int index = getIndex(threadsPerBlock);

    // each kernel will update the curent row of the problem
    if (index < problemSize) {

        for (int i = 0; i < problemSize; i++) {
            pheromoneMatrix[problemSize * index + i] = pheromoneMatrix[problemSize * index + i] * PHEROMONE_EVAPORATION_RATE;
        }
    }
}

__global__ void distanceReset(double* distanceHistory, int numAnts, int problemSize, int threadsPerBlock) {
        for (int i = 0; i < numAnts; i++) {
            distanceHistory[i] = 0;
        }
    

}
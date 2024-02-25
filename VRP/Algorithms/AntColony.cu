
#include "AntColony.cuh"

#include <iostream>

Ant* host_ants;
Ant* dev_ants;
int cities;

double* host_distances, *host_pheromones;
double* dev_distances, *dev_pheromones;


double* host_bestDistances;
double* dev_bestDistances;

__device__ unsigned int get_index()
{
    return blockDim.x * blockIdx.x + threadIdx.x;
}

void allocate_values()
{
    host_distances = (double*)(malloc(cities * cities * sizeof(double)));
    host_pheromones = (double*)(malloc(cities * cities * sizeof(double)));
    host_bestDistances = (double*)(malloc(NUM_ANTS * sizeof(double)));

    cudaMalloc(&dev_distances, cities * cities * sizeof(double));
    cudaMalloc(&dev_pheromones, cities * cities * sizeof(double));
    cudaMalloc(&dev_bestDistances, NUM_ANTS * sizeof(double));

}

void initialize_values()
{
    //randomly initialize distance matrix
    for(int i = 0; i < cities; i++)
    {
        for(int j = 0; j < cities; j++)
        {
            int index = i * cities + j;
            if(i != j)
            {
                host_distances[index] = (double)(rand() % 100) + 1.0;
            }
            else
            {
                host_distances[index] = 0.0;
            }
        }
    }

    //initialize pheromone matrix to 1.0 as an arbitrary starting level
    for(int i = 0; i < cities * cities; i++)
    {
        host_pheromones[i] = 1.0;
    }

    cudaMemcpy(dev_distances, host_distances, cities * cities * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pheromones, host_pheromones, cities * cities * sizeof(double), cudaMemcpyHostToDevice);
}

void print_matrix(const double* matrix, const char* msg)
{
    printf("%s: \n", msg);
    for(int i = 0; i < cities; i++)
    {
        for(int j = 0; j < cities; j++)
        {
            printf("%2.2f\t", matrix[i * cities + j]);
        }
        printf("\n");
    }
}

void free_values()
{
    free(host_distances);
    free(host_pheromones);
    free(host_bestDistances);

    cudaFree(dev_distances);
    cudaFree(dev_pheromones);
    cudaFree(dev_bestDistances);
}


int ACO_main()
{
    std::cout << "Yo from new" << std::endl;

    cities = 10;

    allocate_values();
    initialize_values();

    print_matrix(host_distances, "Distances");
    print_matrix(host_pheromones, "Pheromones");


    const int matrix_blocks = static_cast<int>(floor((cities * cities) / THREADS_PER_BLOCK)) + 1;
    for(int iter = 0; iter < NUM_ITERATIONS; iter++)
    {
        evaporate_pheromone<<<matrix_blocks, THREADS_PER_BLOCK>>>(dev_pheromones, cities);

        //just for printing, delete eventually
        cudaMemcpy(host_pheromones, dev_pheromones, cities * cities * sizeof(double), cudaMemcpyDeviceToHost);
        print_matrix(host_pheromones, "Pheromones");
    }

    /*
    host_ants = (Ant*)(malloc(sizeof(Ant) * NUM_ANTS));
    for(int i = 0; i < NUM_ANTS; i++)
    {
        Ant a;
        a.tour = (int*)(malloc(sizeof(int) * cities));
        for(int j = 0; j < cities; j++)
        {
            a.tour[j] = cities - j;
        }
        a.current_city = i;
        host_ants[i] = a;
        //free(a->tour);
        //free(a);
    }

    cudaMalloc((void**)&dev_ants, sizeof(Ant) * NUM_ANTS);
    for(int i = 0; i < NUM_ANTS; i++)
    {
        int* dev_tour;
        cudaMalloc((void**)(&dev_tour), sizeof(int) * cities);
        cudaMemcpy(dev_tour, host_ants[i].tour, sizeof(int) * cities, cudaMemcpyHostToDevice);

        Ant temp_dev_ant = {host_ants[i].current_city, dev_tour, host_ants[i].distance};
        cudaMemcpy(&dev_ants[i], &temp_dev_ant, sizeof(Ant), cudaMemcpyHostToDevice);
    }

    Test_PrintInitialAnts<<<GROUPS_OF_N_ANTS, THREADS_PER_BLOCK>>>(dev_ants);
    cudaDeviceSynchronize();

    for(int i = 0; i < NUM_ANTS; i++)
    {
        free(host_ants[i].tour);
    }

    free(host_ants);
    cudaFree(dev_ants);

    free(host_distances);
    free(host_pheromones);
     */

    free_values();
    cudaDeviceReset();
    return EXIT_SUCCESS;
}

__global__ void evaporate_pheromone(double *pheromones, int num_cities)
{
    unsigned int index = get_index();
    if(index < num_cities * num_cities)
    {
        pheromones[index] *= (1.0 - RHO);
    }
}
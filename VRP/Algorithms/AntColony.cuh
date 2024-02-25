#ifndef ant_colony_cu
#define ant_colony_cu

#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../ModelParameters.h"


typedef struct
{
    int current_city;
    int* tour;
    double distance;
} Ant;


int ACO_main();

__global__ void ACOPrint();

__global__ void init_ants();

__global__ void
move_ant(const double *matrix_pheromones, const double *matrix_distances, curandStateXORWOW *states,
         double *history_distances, int *history_tours, int *history_visited, int num_cities);
__global__ void evaporate_pheromone(double *matrix_pheromones, int num_cities);
__global__ void update_pheromone(double* matrix_pheromones, const double* history_distances, const int* history_tours, int num_cities);
__global__ void reset_histories(double* history_distances, int* history_tours, int* history_visited, int num_cities);

__global__ void Test_PrintInitialAnts(const Ant* ants);


//double*

#endif //ant_colony_cu
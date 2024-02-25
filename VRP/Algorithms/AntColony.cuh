#ifndef ant_colony_cu
#define ant_colony_cu

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

__global__ void evaporate_pheromone(double *pheromones, int num_cities);
__global__ void update_pheromone();
__global__ void move_ant();

__global__ void Test_PrintInitialAnts(const Ant* ants);


//double*

#endif //ant_colony_cu
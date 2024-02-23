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


__global__ void ACOPrint();

#endif //ant_colony_cu
#ifndef ant_colony_cu
#define ant_colony_cu

#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../ModelParameters.h"

//main function
double ACO_main(double *adj_mat, int size);

//kernel calls
__global__ void
move_ant(const double *matrix_pheromones, const double *matrix_distances, curandStateXORWOW *states,
         double *history_distances, int *history_tours, int *history_visited, int num_cities);
__global__ void evaporate_pheromone(double *matrix_pheromones, int num_cities);
__global__ void update_pheromone(double* matrix_pheromones, const double* history_distances, const int* history_tours, int num_cities);
__global__ void reset_histories(double* history_distances, int* history_tours, int* history_visited, int num_cities);

//CUDA helper functions
__device__ unsigned int get_index();
__global__ void setup_curand_states(curandState* dev_states, unsigned long seed);
void print_matrix(const double* matrix, const char* msg);

//host memory allocation and deallocation functions
void allocate_memory();
void initialize_values(double *adj_mat);
void free_memory();

#endif //ant_colony_cu
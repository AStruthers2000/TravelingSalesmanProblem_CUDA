
#include "AntColony.cuh"

__global__ void ACOPrint()
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from ACO print thread %d in block %d. I am index %d. There will be %d ants total\n", threadIdx.x, blockIdx.x, index, NUM_ANTS);

    Ant a;
    a.current_city = 0;
    a.tour[0] = a.current_city;
    a.distance = 0;

}


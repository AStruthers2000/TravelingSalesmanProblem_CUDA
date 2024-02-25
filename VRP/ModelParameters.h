#ifndef ModelParameters
#define ModelParameters

#define THREADS_PER_BLOCK 32                                //Should be some power of 2, no greater than cudaDeviceProp.maxThreadsPerBlock
#define GROUPS_OF_N_ANTS 1                                  //Number of groups of THREADS_PER_BLOCK ants. Recommended to be >= to cudaDeviceProp.multiProcessorCount to fully utilize GPU
#define NUM_ANTS (GROUPS_OF_N_ANTS * THREADS_PER_BLOCK)     //Total number of ants (blocks * threads per block)

#define ALPHA 0.25                                          //Pheromone importance
#define BETA 2.0                                            //Heuristic importance
#define RHO 0.5                                             //Evaporation rate
#define Q 100.0                                             //Pheromone deposit factor

#define NUM_ITERATIONS 1000                                 //Number of iterations for ants to run

#endif //ModelParameters
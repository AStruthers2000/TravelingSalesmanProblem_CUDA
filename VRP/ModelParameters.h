#ifndef ModelParameters
#define ModelParameters

#define THREADS_PER_BLOCK 1024                              //Should be some power of 2, no greater than cudaDeviceProp.maxThreadsPerBlock
#define GROUPS_OF_N_ANTS 10                                 //Number of groups of THREADS_PER_BLOCK ants. Recommended to be >= to cudaDeviceProp.multiProcessorCount to fully utilize GPU
#define NUM_ANTS (GROUPS_OF_N_ANTS * THREADS_PER_BLOCK)     //Total number of ants (blocks * threads per block)



#endif //ModelParameters
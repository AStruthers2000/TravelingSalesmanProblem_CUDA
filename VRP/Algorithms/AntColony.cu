
#include "AntColony.cuh"

#include <iostream>
#include <curand_kernel.h>

/********** Global memory **********/
int cities;

double* host_distances, *host_pheromones;
double* dev_distances, *dev_pheromones;

double* host_distancesHistory;
double* dev_distancesHistory;

int* dev_toursHistory, *dev_visitedHistory;

curandState* dev_curandStates;

/********** Main function **********/
/**
 *
 * @param adj_mat
 * @param size
 * @return
 */
double ACO_main(double *adj_mat, int size)
{
    cities = size;

    allocate_memory();
    initialize_values(adj_mat);

    //print_matrix(host_distances, "Distances");
    //print_matrix(host_pheromones, "Pheromones");

    time_t seed;
    time(&seed);

    setup_curand_states<<<GROUPS_OF_N_ANTS, THREADS_PER_BLOCK>>>(dev_curandStates, (unsigned long) seed);
    cudaDeviceSynchronize();


    const int matrix_blocks = static_cast<int>(floor((cities * cities) / THREADS_PER_BLOCK)) + 1;
    for(int iter = 0; iter < NUM_ITERATIONS; iter++)
    {
        move_ant<<<GROUPS_OF_N_ANTS, THREADS_PER_BLOCK>>>(dev_pheromones, dev_distances, dev_curandStates, dev_distancesHistory,
                                                          dev_toursHistory, dev_visitedHistory, cities);
        cudaDeviceSynchronize();

//        auto host_dist = (double*)(malloc(NUM_ANTS * sizeof(double)));
//        cudaMemcpy(host_dist, dev_distancesHistory, NUM_ANTS * sizeof(double), cudaMemcpyDeviceToHost);
//        double sum = 0.0;
//        double min = std::numeric_limits<double>::max();
//        for(int i = 0; i < NUM_ANTS; i++)
//        {
//            sum += host_dist[i];
//            if(host_dist[i] < min)
//                min = host_dist[i];
//        }
//        double avg = sum / NUM_ANTS;
//        std::cout << "Average value on iter " << iter << ": " << avg << std::endl;
//        std::cout << "Minimum value on iter " << iter << ": " << min << std::endl;
//        free(host_dist);

        evaporate_pheromone<<<matrix_blocks, THREADS_PER_BLOCK>>>(dev_pheromones, cities);
        cudaDeviceSynchronize();
        //cudaMemcpy(host_pheromones, dev_pheromones, cities * cities * sizeof(double), cudaMemcpyDeviceToHost);
        //print_matrix(host_pheromones, "Pheromones after evaporating");

        update_pheromone<<<GROUPS_OF_N_ANTS, THREADS_PER_BLOCK>>>(dev_pheromones, dev_distancesHistory,
                                                                      dev_toursHistory, cities);
        cudaDeviceSynchronize();
        //cudaMemcpy(host_pheromones, dev_pheromones, cities * cities * sizeof(double), cudaMemcpyDeviceToHost);
        //print_matrix(host_pheromones, "Pheromones after updating");

        if(iter < NUM_ITERATIONS - 1)
        {
            reset_histories<<<GROUPS_OF_N_ANTS, THREADS_PER_BLOCK>>>(dev_distancesHistory, dev_toursHistory,
                                                                     dev_visitedHistory, cities);
            cudaDeviceSynchronize();
        }
        //std::cout << "Iteration " << iter << " completed" << std::endl;
        //std::cout << "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-" << std::endl;
    }


    auto host_dist = (double*)(malloc(NUM_ANTS * sizeof(double)));
    cudaMemcpy(host_dist, dev_distancesHistory, NUM_ANTS * sizeof(double), cudaMemcpyDeviceToHost);
    double min = std::numeric_limits<double>::max();
    for(int i = 0; i < NUM_ANTS; i++)
    {
        if(host_dist[i] < min)
            min = host_dist[i];
    }
    std::cout << "Minimum value: " << min << std::endl;
    free(host_dist);

    free_memory();
    cudaDeviceReset();
    return min;
}


/********** Kernel calls **********/
/**
 *
 * @param matrix_pheromones
 * @param matrix_distances
 * @param states
 * @param history_distances
 * @param history_tours
 * @param history_visited
 * @param num_cities
 */
__global__ void
move_ant(const double *matrix_pheromones, const double *matrix_distances, curandState *states,
         double *history_distances, int *history_tours, int *history_visited, int num_cities)
{
    //where the ant's history starts
    unsigned int ant_id = get_index();

    //we will get an even distribution of ants starting at the various cities
    //with roughly an even number of ants starting at each city
    int current_city = static_cast<int>(ant_id) % num_cities;
    history_visited[ant_id * num_cities + current_city] = 1;
    history_tours[ant_id * num_cities] = current_city;

    //printf("Thread %d:%d starts at city %d\n", blockIdx.x, threadIdx.x, current_city);

    //we will iterate n - 1 times, because we already know where we want to start
    //on each iteration, we will calculate the probabilities of visiting each city
    //that this ant hasn't visited yet, then use roulette wheel selection to visit
    //a city determined probabilistically. Once this loop completes, we need to also
    //add the distance travelled from the last visited city, back to the beginning
    //because a tour is only finished once we return to where we started
    for(int i = 1; i < num_cities; i++)
    {
        //we only care about accumulating probability for cities we have not visited
        //we want to compute:
        //\sum_{m\in allowed} \tau_{im}^\alpha \cdot \eta_{im}^\beta
        //where allowed is all cities we haven't visited yet
        double total_prob = 0.0;
        for(int next_city = 0; next_city < num_cities; next_city++)
        {
            //aka we have not yet visited this city
            if (history_visited[ant_id * num_cities + next_city] != 1)
            {
                int city_index = current_city * num_cities + next_city;
                double tau = matrix_pheromones[city_index];
                double eta = 1.0 / matrix_distances[city_index];
                total_prob += pow(tau, ALPHA) * pow(eta, BETA);
            }
        }

        //perform roulette wheel selection to select our next city
        double r = curand_uniform(&states[ant_id]) * total_prob;
        double accum_prob = 0.0;
        int selected_city = -1;

        //printf("Thread %d:%d generated %2.2f\n", blockIdx.x, threadIdx.x, r);
        //for each possible city, we want to compute:
        //\tau_{im}^\alpha \cdot \eta_{im}^\beta
        //so that we can calculate P_{ij} as this product divided by the sum calculated above
        int last_best = 0;
        for(int next_city = 0; next_city < num_cities; next_city++)
        {
            if(history_visited[ant_id * num_cities + next_city] != 1)
            {
                last_best = next_city;

                int city_index = current_city * num_cities + next_city;

                double tau = matrix_pheromones[city_index];
                double eta = 1.0 / matrix_distances[city_index];

                double prob = pow(tau, ALPHA) * pow(eta, BETA);
                //double prob = num / total_prob;
                accum_prob += prob;
                //printf("\tThread %d:%d tried to go to city %d, with a probability of %2.2f where accum prob = %2.2f\n", blockIdx.x, threadIdx.x, next_city, prob, accum_prob);

                if(accum_prob >= r)
                {
                    selected_city = next_city;
                    break;
                }
            }
        }

        if(selected_city == -1)
        {
            selected_city = last_best;
            printf("ERROR -1 || %d || accumProb %f || r %f\n", ant_id, accum_prob, r);
        }

        //printf("Thread %d:%d is now visiting city %d\n", blockIdx.x, threadIdx.x, selected_city);

        //now that we have selected a city using the probability equation
        //we want to actually move to that city.
        //We need to add this city to this ant's tour, add this city to this
        //ant's visited history, and add the distance from this move to this ant's
        //distance history
        history_tours[ant_id * num_cities + i] = selected_city;
        history_visited[ant_id * num_cities + selected_city] = 1;
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

/**
 *
 * @param matrix_pheromones
 * @param num_cities
 */
__global__ void evaporate_pheromone(double *matrix_pheromones, int num_cities)
{
    unsigned int index = get_index();
    if(index < num_cities * num_cities)
    {
        //printf("Evaporate that shit\n");
        matrix_pheromones[index] *= (1.0 - RHO);

        //never let the pheromone trail get to 0 or less, because we want to encourage exploration
        //if the pheromone trail was = 0, no ant would ever walk on this route again
        //if the pheromone trail was < 0, that doesn't make sense. ants don't produce ANTi-pheromone (ha, ant pun, thanks Abbi)

        /*
        if(matrix_pheromones[index] <= 0.0)
        {
            matrix_pheromones[index] = 1.0 / num_cities;
        }
         */
    }
}

/**
 *
 * @param matrix_pheromones
 * @param history_distances
 * @param history_tours
 * @param num_cities
 */
__global__ void update_pheromone(double* matrix_pheromones, const double* history_distances, const int* history_tours, int num_cities)
{
    unsigned int ant_id = get_index();

    //assert(history_distances[ant_id] > 0);
    double pheromone_delta = Q / (history_distances[ant_id] + 0.001);
    for(int i = 0; i < num_cities; i++)
    {
        int from, to;

        //if we aren't at the end of the tour yet
        if(i < num_cities - 1)
        {
            //we want to add the pheromone delta to the edge {v_i, v_{i+1}}
            from = history_tours[ant_id * num_cities + i];
            to = history_tours[ant_id * num_cities + i + 1];
        }
        //if we are at the end of the tour
        else
        {
            //we want to add the pheromone delta to the edge {v_{n-1}, v_0}
            //this is because we want to end the full tour where we started (v_0)
            from = history_tours[ant_id * num_cities + i];
            to = history_tours[ant_id * num_cities];
        }

        //we update the entry at city_index with the pheromone delta
        //because we are using the pheromone matrix to represent the weight
        //of traveling where start = from and end = to. Since we collapsed the
        //matrix into a 1d array, the edge between vertices {from, to} is stored
        //at the ant_id from * n + to, where n = num_cities. This is a standard way
        //of using a 1d array as a matrix
        int city_index = from * num_cities + to;

        //printf("Ant %d went from %d to %d\n", ant_id, from, to);

        //this kernel runs once for each ant, so that each ant's matrix_pheromones
        //can be calculated in parallel, but we must use atomic operations
        //as to not overwrite our update when another ant is currently updating
        auto a = &matrix_pheromones[city_index];
        atomicAdd(a, pheromone_delta);
    }
}

/**
 *
 * @param history_distances
 * @param history_tours
 * @param history_visited
 * @param num_cities
 */
__global__ void reset_histories(double *history_distances, int *history_tours, int *history_visited, int num_cities)
{
    unsigned int ant_id = get_index();
    history_distances[ant_id] = 0.0;

    for(int i = 0; i < num_cities; i++)
    {
        history_tours[ant_id * num_cities + i] = 0;
        history_visited[ant_id * num_cities + i] = 0;
    }
}


/********** Memory allocation and deallocation **********/
void allocate_memory()
{
    host_distances = (double*)(malloc(cities * cities * sizeof(double)));
    host_pheromones = (double*)(malloc(cities * cities * sizeof(double)));
    host_distancesHistory = (double*)(malloc(NUM_ANTS * sizeof(double)));

    cudaMalloc(&dev_distances, cities * cities * sizeof(double));
    cudaMalloc(&dev_pheromones, cities * cities * sizeof(double));
    cudaMalloc(&dev_distancesHistory, NUM_ANTS * sizeof(double));
    cudaMalloc(&dev_toursHistory, NUM_ANTS * cities * sizeof(int));
    cudaMalloc(&dev_visitedHistory, NUM_ANTS * cities * sizeof(int));
    cudaMalloc(&dev_curandStates, NUM_ANTS * sizeof(curandState));
}

void initialize_values(double *adj_mat)
{
    memcpy(host_distances, adj_mat, cities * cities * sizeof(double));
    //print_matrix(host_distances, "Distances:");

    //initialize pheromone matrix to INIT_PHEROMONE_LEVEL as an arbitrary starting level
    for(int i = 0; i < cities; i++)
    {
        for(int j = 0; j < cities; j++)
        {
            int index = i * cities + j;
            if (i != j)
            {
                host_pheromones[index] = INIT_PHEROMONE_LEVEL;
            }
            else
            {
                host_pheromones[index] = 0.0;
            }
        }
    }

    cudaMemcpy(dev_distances, host_distances, cities * cities * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pheromones, host_pheromones, cities * cities * sizeof(double), cudaMemcpyHostToDevice);
}

void free_memory()
{
    free(host_distances);
    free(host_pheromones);
    free(host_distancesHistory);

    cudaFree(dev_distances);
    cudaFree(dev_pheromones);
    cudaFree(dev_distancesHistory);
    cudaFree(dev_toursHistory);
    cudaFree(dev_visitedHistory);
    cudaFree(dev_curandStates);
}


/********** Helper functions **********/
__device__ unsigned int get_index()
{
    return blockDim.x * blockIdx.x + threadIdx.x;
}

__global__ void setup_curand_states(curandState* dev_states, unsigned long seed)
{
    unsigned int index = get_index();
    if(index < NUM_ANTS)
    {
        curand_init(seed, index, 0, &dev_states[index]);
    }
}

[[maybe_unused]] void print_matrix(const double* matrix, const char* msg)
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


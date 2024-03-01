
#include "AntColony.cuh"

#include <iostream>
#include <curand_kernel.h>

//Ant* host_ants;
//Ant* dev_ants;
int cities;

double* host_distances, *host_pheromones;
double* dev_distances, *dev_pheromones;

double* host_distancesHistory;
double* dev_distancesHistory;

int* dev_toursHistory, *dev_visitedHistory;

curandState* dev_curandStates;

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

void initialize_values()
{
    //randomly initialize distance matrix

    double cheat[841] = {0, 107, 241, 190, 124, 80, 316, 76, 152, 157, 283, 133, 113, 297, 228, 129, 348, 276, 188, 150, 65, 341, 184, 67, 221, 169, 108, 45, 167,
                     107, 0, 148, 137, 88, 127, 336, 183, 134, 95, 254, 180, 101, 234, 175, 176, 265, 199, 182, 67, 42, 278, 271, 146, 251, 105, 191, 139, 79,
                     241, 148, 0, 374, 171, 259, 509, 317, 217, 232, 491, 312, 280, 391, 412, 349, 422, 356, 355, 204, 182, 435, 417, 292, 424, 116, 337, 273, 77,
                     190, 137, 374, 0, 202, 234, 222, 192, 248, 42, 117, 287, 79, 107, 38, 121, 152, 86, 68, 70, 137, 151, 239, 135, 137, 242, 165, 228, 205,
                     124, 88, 171, 202, 0, 61, 392, 202, 46, 160, 319, 112, 163, 322, 240, 232, 314, 287, 238, 155, 65, 366, 300, 175, 307, 57, 220, 121, 97,
                     80, 127, 259, 234, 61, 0, 386, 141, 72, 167, 351, 55, 157, 331, 272, 226, 362, 296, 232, 164, 85, 375, 249, 147, 301, 118, 188, 60, 185,
                     316, 336, 509, 222, 392, 386, 0, 233, 438, 254, 202, 439, 235, 254, 210, 187, 313, 266, 154, 282, 321, 298, 168, 249, 95, 437, 190, 314, 435,
                     76, 183, 317, 192, 202, 141, 233, 0, 213, 188, 272, 193, 131, 302, 233, 98, 344, 289, 177, 216, 141, 346, 108, 57, 190, 245, 43, 81, 243,
                     152, 134, 217, 248, 46, 72, 438, 213, 0, 206, 365, 89, 209, 368, 286, 278, 360, 333, 284, 201, 111, 412, 321, 221, 353, 72, 266, 132, 111,
                     157, 95, 232, 42, 160, 167, 254, 188, 206, 0, 159, 220, 57, 149, 80, 132, 193, 127, 100, 28, 95, 193, 241, 131, 169, 200, 161, 189, 163,
                     283, 254, 491, 117, 319, 351, 202, 272, 365, 159, 0, 404, 176, 106, 79, 161, 165, 141, 95, 187, 254, 103, 279, 215, 117, 359, 216, 308, 322,
                     133, 180, 312, 287, 112, 55, 439, 193, 89, 220, 404, 0, 210, 384, 325, 279, 415, 349, 285, 217, 138, 428, 310, 200, 354, 169, 241, 112, 238,
                     113, 101, 280, 79, 163, 157, 235, 131, 209, 57, 176, 210, 0, 186, 117, 75, 231, 165, 81, 85, 92, 230, 184, 74, 150, 208, 104, 158, 206,
                     297, 234, 391, 107, 322, 331, 254, 302, 368, 149, 106, 384, 186, 0, 69, 191, 59, 35, 125, 167, 255, 44, 309, 245, 169, 327, 246, 335, 288,
                     228, 175, 412, 38, 240, 272, 210, 233, 286, 80, 79, 325, 117, 69, 0, 122, 122, 56, 56, 108, 175, 113, 240, 176, 125, 280, 177, 266, 243,
                     129, 176, 349, 121, 232, 226, 187, 98, 278, 132, 161, 279, 75, 191, 122, 0, 244, 178, 66, 160, 161, 235, 118, 62, 92, 277, 55, 155, 275,
                     348, 265, 422, 152, 314, 362, 313, 344, 360, 193, 165, 415, 231, 59, 122, 244, 0, 66, 178, 198, 286, 77, 362, 287, 228, 358, 299, 380, 319,
                     276, 199, 356, 86, 287, 296, 266, 289, 333, 127, 141, 349, 165, 35, 56, 178, 66, 0, 112, 132, 220, 79, 296, 232, 181, 292, 233, 314, 253,
                     188, 182, 355, 68, 238, 232, 154, 177, 284, 100, 95, 285, 81, 125, 56, 66, 178, 112, 0, 128, 167, 169, 179, 120, 69, 283, 121, 213, 281,
                     150, 67, 204, 70, 155, 164, 282, 216, 201, 28, 187, 217, 85, 167, 108, 160, 198, 132, 128, 0, 88, 211, 269, 159, 197, 172, 189, 182, 135,
                     65, 42, 182, 137, 65, 85, 321, 141, 111, 95, 254, 138, 92, 255, 175, 161, 286, 220, 167, 88, 0, 299, 229, 104, 236, 110, 149, 97, 108,
                     341, 278, 435, 151, 366, 375, 298, 346, 412, 193, 103, 428, 230, 44, 113, 235, 77, 79, 169, 211, 299, 0, 353, 289, 213, 371, 290, 379, 332,
                     184, 271, 417, 239, 300, 249, 168, 108, 321, 241, 279, 310, 184, 309, 240, 118, 362, 296, 179, 269, 229, 353, 0, 121, 162, 345, 80, 189, 342,
                     67, 146, 292, 135, 175, 147, 249, 57, 221, 131, 215, 200, 74, 245, 176, 62, 287, 232, 120, 159, 104, 289, 121, 0, 154, 220, 41, 93, 218,
                     221, 251, 424, 137, 307, 301, 95, 190, 353, 169, 117, 354, 150, 169, 125, 92, 228, 181, 69, 197, 236, 213, 162, 154, 0, 352, 147, 247, 350,
                     169, 105, 116, 242, 57, 118, 437, 245, 72, 200, 359, 169, 208, 327, 280, 277, 358, 292, 283, 172, 110, 371, 345, 220, 352, 0, 265, 178, 39,
                     108, 191, 337, 165, 220, 188, 190, 43, 266, 161, 216, 241, 104, 246, 177, 55, 299, 233, 121, 189, 149, 290, 80, 41, 147, 265, 0, 124, 263,
                     45, 139, 273, 228, 121, 60, 314, 81, 132, 189, 308, 112, 158, 335, 266, 155, 380, 314, 213, 182, 97, 379, 189, 93, 247, 178, 124, 0, 199,
                     167, 79, 77, 205, 97, 185, 435, 243, 111, 163, 322, 238, 206, 288, 243, 275, 319, 253, 281, 135, 108, 332, 342, 218, 350, 39, 263, 199, 0};

    /*
    double cheat[25] = {0, 30.4138126514911, 46.09772228646444, 48.25971404805462, 37.53664875824692,
                        30.4138126514911, 0, 30.0, 49.73932046178355, 59.61543424315552,
                        46.09772228646444, 30.0, 0, 28.178005607210743, 55.44366510251645,
                        48.25971404805462, 49.73932046178355, 28.178005607210743, 0, 36.05551275463989,
                        37.53664875824692, 59.61543424315552, 55.44366510251645, 36.05551275463989, 0, };
                        */
    for(int i = 0; i < cities; i++)
    {
        for(int j = 0; j < cities; j++)
        {
            int index = i * cities + j;
            host_distances[index] = cheat[index];
            /*
            if(i != j)
            {
                host_distances[index] = (double)(rand() % 100) + 1.0;
            }
            else
            {
                host_distances[index] = 0.0;
            }
             */
        }
    }

    //initialize pheromone matrix to 1.0 as an arbitrary starting level
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


double ACO_main()
{
    cities = 29;

    allocate_memory();
    initialize_values();

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

    free_memory();
    cudaDeviceReset();
    return min;
}

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



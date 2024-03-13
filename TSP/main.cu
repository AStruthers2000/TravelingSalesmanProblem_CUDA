/**
 * \mainpage
 * The primary objective of this project is to explore the potential of Ant Colony Optimization (ACO), a metaheuristic
 * inspired by the foraging behavior of ants, in solving the Travelling Salesman Problem (TSP) using Compute Unified Device Architecture (CUDA).
 * The goal of this project is to demonstrate the feasibility and advantages of using GPGPU-based parallel computing in
 * solving complex combinatorial optimization problems.  By harnessing the parallel processing capabilities of CUDA,
 * we seek to develop an efficient parallel implementation of ACO for TSP solving on GPGPUs, aiming to significantly
 * reduce the computational time required to find near-optimal solutions for large-scale instances of the problem.
 *
 * The Traveling Salesman Problem (TSP) is one of the most extensively studied combinatorial optimization problems in
 * computer science and logistics and operations research. The classic algorithmic problem is a standard benchmark for
 * evaluating the efficacy of various combinatorial algorithms. In this problem, a salesman is given a list of cities
 * and must determine the shortest route that allows him to visit each city once and return to his original location.
 * The TSP is seemingly simple in problem statement yet the complexity required in finding the optimal solution is what
 * makes it such a studied problem. It is an NP-hard problem in combinatorial optimization, important in operations
 * research and theoretical computer science. The TSP requires finding the shortest possible tour that visits each city
 * in a network of cities exactly once while returning to the starting city. This problem and its many variants are
 * applicable in logistics, transportation planning, and network design, among others.
 *
 * Compute Unified Device Architecture (CUDA) is a parallel computing platform and application programming interface
 * (API) model created by NVIDIA. It allows software developers to use a CUDA-enabled graphics processing unit (GPU)
 * for general purpose processing, an approach known as GPGPU (General-Purpose computing on Graphics Processing Units).
 * By utilizing CUDA, programmers can offload computationally intensive tasks from the CPU to the GPGPU, exploiting the
 * massive parallelism inherent in GPGPU architectures to accelerate the execution of algorithms. CUDA provides a
 * significant increase in computing performance by harnessing the power of the graphics processing unit (GPU). With
 * millions of CUDA-capable GPGPUs sold to date, software developers, scientists, and researchers are finding
 * broad-ranging uses for GPGPU computing with CUDA.
 *
 * This project presents a unique intersection of combinatorial optimization, parallel computing, and artificial
 * intelligence. Our primary objective is to develop a high-performance parallel ACO implementation capable of
 * efficiently solving large-scale instances of the TSP by exploiting the computational prowess of GPGPUs. The TSP,
 * a classic NP-hard problem in combinatorial optimization, challenges researchers with finding the shortest route that
 * visits a set of cities exactly once before returning to the origin city. Traditional exact algorithms often falter
 * when faced with the exponential explosion of possible solutions, prompting the exploration of heuristic and
 * metaheuristic techniques like ACO. Concurrently, CUDA serves as a formidable tool for parallel computing, offering us
 * a platform to tap into the immense parallel processing capabilities of modern GPGPUs.
 */

//system includes
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <string>
#include <regex>
#include <filesystem>

//our includes
#include "Algorithms/AntColony.cuh"

using namespace std;
using namespace chrono;

constexpr char WRITE_FILENAME[256] = R"(.\TSP_Output.txt)";

void load_adjacency_matrix_from_file(const string &filename, int size, double *&adj_mat);
void write_to_file(const string& problem, int experiment, double result, double execution_time);
vector<string> split_string_by_token(const string& line, char token);

/**
 * \brief Main entry point for the algorithm. Accepts two command line arguments.
 *
 * The arguments provided are intended to be integer numbers, where the first argument is the lower bound of the problem instance
 * and the second argument is the upper bound. An example use case for this is to say "vrp.exe 10 100" which will solve all
 * problems stored in the \Datasets\ folder that have a size between 10 cities and 100 cities. Additionally, only one argument
 * could be passed, such as "vrp.exe 1000" which will solve all problems that have a size of >= 1000 cities. If no arguments
 * are provided, the default behavior is to try all problems in the file. This range is inclusive, meaning if the user passes
 * "vrp.exe 1092 1092", all problems of size 1092 will be attempted.
 * @param argc Number of arguments. Supported number of additional arguments is either 0, 1, or 2
 * @param argv String array of all passed arguments. Argument 0 is always the EXE name. Argument 1 and argument 2 will be read as integers0
 * @return EXIT_SUCCESS on successful completion of solving the given problems
 */
int main(int argc, char** argv)
{
    string path = R"(.\Datasets\)";

    //track the upper and lower bounds on the problem instances we want to solve
    int lower_size = 0;
    int upper_size = numeric_limits<int>::max();

    //if 3 arguments were passed (exe name and 2 numbers), read them as lower and upper limits
    if(argc >= 3)
    {
        lower_size = stoi(argv[1]);
        upper_size = stoi(argv[2]);
    }
    //if only 2 arguments were passed, read the 2nd as the lower limit
    else if(argc == 2)
    {
        lower_size = stoi(argv[1]);
    }

    namespace fs = std::filesystem;

    //regex pattern to read the numbers in the filename, useful for finding the problem size before parsing the data
    regex size_regex("\\d+");
    for (const auto& entry : fs::directory_iterator(path))
    {
        // entry.path() returns the full path of the file
        const string file = entry.path().filename().generic_string();
        const string stripped_file = split_string_by_token(file, '.')[0];
        smatch match;

        if (regex_search(file, match, size_regex))
        {
            //read the part of the filename that is only numbers as an int
            int size = stoi(match.str());

            //skip this problem if its size is outside the problem bounds
            if(size < lower_size || size > upper_size) continue;

            //load the adjacency matrix into memory as a flattened 1D array
            auto adj_mat = (double *) malloc(size * size * sizeof(double));
            load_adjacency_matrix_from_file(entry.path().generic_string(), size, adj_mat);

            cout << "Starting experimentation on problem " << stripped_file << endl;

            int experiments = 30;
            double sols = 0.0;
            double min = numeric_limits<double>::max();
            auto start = high_resolution_clock::now();

            //perform experiments number of independent experiments on the given problem
            for (int i = 0; i < experiments; i++)
            {
                auto problem_start = high_resolution_clock::now();

                //call the ACO solver with the problem definition (adjacency matrix) and problem size
                double sol = ACO_main(adj_mat, size);

                auto problem_end = high_resolution_clock::now();
                auto problem_duration = duration_cast<microseconds>(problem_end - problem_start).count();

                write_to_file(stripped_file, i, sol, static_cast<double>(problem_duration) / 1e06);

                sols += sol;
                if (sol < min)
                {
                    min = sol;
                }
            }

            auto end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end - start).count();
            double avg = sols / experiments;

            printf("Best solution for %d experiments:    \t%2.4f\n", experiments, min);
            printf("Average solution for %d experiments: \t%2.4f\n", experiments, avg);
            printf("Took %lld seconds for all %d experiments\n\tAverage of %lld milliseconds per experiment\n",
                   duration / static_cast<long long>(1e06), experiments, (duration / 1000) / experiments);
            printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n");

            //free our problem. If we malloc memory, we also need to free memory!!!!
            free(adj_mat);
        }
    }
    return EXIT_SUCCESS;
}
/********** Helper functions **********/

/**
 * \brief Helper function to read the adjacency matrix of a problem to a 1D array
 *
 * Takes a filename and a problem size, then reads the adjacency matrix stored in that file while loading it into
 * preallocated memory.
 * @param filename Name of the file containing the problem matrix
 * @param size Size of the problem, aka the dimensions of the matrix
 * @param adj_mat out parameter as a pointer reference so that we can write directly to this memory instead of needing multiple mallocs
 */
void load_adjacency_matrix_from_file(const string &filename, int size, double *&adj_mat)
{
    ifstream file(filename);
    if (file.is_open())
    {
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                file >> adj_mat[i * size + j];
            }
        }
        file.close();
    } else
    {
        cerr << "Unable to open file: " << filename << endl;
    }
}

/**
 * \brief Simple file writing function to log experimental results
 *
 * Takes a problem name and some basic information to be appended to a data file. This file is in CSV format.
 * @param problem Name of the problem this information is associated with
 * @param experiment Experiment number to track the number of experiments performed and the percent completion of each problem instance
 * @param result The best found solution by any ant over all iterations
 * @param execution_time The number of seconds the code was executing to find the given solution
 */
void write_to_file(const string& problem, int experiment, double result, double execution_time)
{
    ofstream file;
    file.open(WRITE_FILENAME, ios_base::app);
    if(!file.is_open())
    {
        cerr << "Failed to write to output file: " << WRITE_FILENAME << endl;
    }

    file << problem << ",";
    file << experiment << ",";
    file << result << ",";
    file << execution_time;

    file << "\n";
    file.close();
}

/**
 * \brief Helper function to split a string by a given char token
 *
 * This helper function is a simple way to tokenize a string into a vector. This is used to strip the file extensions
 * and other useless information from the filename so that the names of problems and problem sizes can be preserved without
 * reporting useless information
 * @param line String line to be tokenized
 * @param token Char token to split the line by
 * @return A vector of all tokens generated by splitting the string by the given token
 */
vector<string> split_string_by_token(const string& line, const char token)
{
    vector<string> split;
    string word;
    istringstream iss(line);
    while(getline(iss, word, token))
    {
        split.push_back(word);
    }
    return split;
}
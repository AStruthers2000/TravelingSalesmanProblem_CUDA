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


int main()
{
    string path = R"(.\Datasets\)";

    namespace fs = std::filesystem;
    vector<string> filenames;
    vector<pair<vector<double>, int>> matrices;
    regex size_regex("\\d+");
    for (const auto& entry : fs::directory_iterator(path))
    {
        // entry.path() returns the full path of the file
        const string file = entry.path().filename().generic_string();
        const string stripped_file = split_string_by_token(file, '.')[0];
        smatch match;

        if (regex_search(file, match, size_regex))
        {
            int size = stoi(match.str());

            auto adj_mat = (double *) malloc(size * size * sizeof(double));
            load_adjacency_matrix_from_file(entry.path().generic_string(), size, adj_mat);

            cout << "Starting experimentation on problem " << stripped_file << endl;

            int experiments = 30;
            double sols = 0.0;
            double min = numeric_limits<double>::max();
            auto start = high_resolution_clock::now();

            for (int i = 0; i < experiments; i++)
            {
                auto problem_start = high_resolution_clock::now();

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

            free(adj_mat);
        }
    }
    return EXIT_SUCCESS;
}
/********** Helper functions **********/

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
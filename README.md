# Traveling Salesman Problem in CUDA
The primary objective of this project is to explore the potential of Ant Colony Optimization (ACO), a metaheuristic inspired by the foraging behavior of ants, in solving the Travelling Salesman Problem (TSP) using Compute Unified Device Architecture (CUDA). The goal of this project is to demonstrate the feasibility and advantages of using GPGPU-based parallel computing in solving complex combinatorial optimization problems.  By harnessing the parallel processing capabilities of CUDA, we seek to develop an efficient parallel implementation of ACO for TSP solving on GPGPUs, aiming to significantly reduce the computational time required to find near-optimal solutions for large-scale instances of the problem.

This project is configured to build using cmake and requires the nvcc NVIDIA CUDA compiler. This is also targeting the CUDA architecture 86 or above, because we use a recent version of the API for atomic operations. We have included a batch file that automatically builds the source into an EXE and moves files around to keep the working directory clean. If you would prefer to compile manually, standard cmake operations should work fine. All of the compiler flags are configured in CMakeLists.txt, which should be sufficient on a CUDA compatible machine.

The EXE accepts one or two additional command line arguments. The following syntax should be used:
1. VRP.exe
2. VRP.exe 10
3. VRP.exe 10 100

In case 1, the EXE runs and all valid TSP instance files stored in the \Datasets\ folder are attempted. In case 2, all problem instances of size >= 10 are run. In case 3, all problem instances of size 10 <= problem size <= 100 are run. This allows you to bound the range of the runs to enable testing of specific files or specific subsets of files. The range is inclusive, so you can run exactly 1 problem instance by specifying "VRP.exe 3038 3038", which will run problem instances of only size = 3038

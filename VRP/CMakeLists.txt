cmake_minimum_required(VERSION 3.27)
project(VRP CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(VRP main.cu)

set_target_properties(VRP PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

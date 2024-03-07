cmake_minimum_required(VERSION 3.27)
project(VRP LANGUAGES CUDA CXX)

find_package(CUDAToolkit)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -g -lineinfo")
set(CMAKE_CUDA_ARCHITECTURES 86)

set(CUDA_SOURCES
        main.cu
        Algorithms/AntColony.cu
        Algorithms/AntColony.cuh
        ModelParameters.h
)

add_executable(VRP ${CUDA_SOURCES})

target_include_directories(VRP PRIVATE Algorithms)

set_target_properties(VRP PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

file(COPY Datasets DESTINATION ${CMAKE_BINARY_DIR})

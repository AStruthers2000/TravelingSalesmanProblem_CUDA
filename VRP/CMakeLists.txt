cmake_minimum_required(VERSION 3.27)
project(VRP LANGUAGES CUDA CXX)

find_package(CUDAToolkit)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -g -lineinfo")

set(CUDA_SOURCES
        main.cu
        Utils/HelperFunctions.cu)

add_executable(VRP ${CUDA_SOURCES})

target_include_directories(VRP PRIVATE ./Utils)
#target_link_libraries(VRP PRIVATE CUDA::curand)

set_target_properties(VRP PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

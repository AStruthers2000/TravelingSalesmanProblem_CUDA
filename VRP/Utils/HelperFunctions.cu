//
// Created by Strut on 2/8/2024.
//
#include "HelperFunctions.cuh"

#include <iostream>
#include <iomanip>

using namespace std;


/*
template <typename T> [[maybe_unused]] __host__ void CustomUtils_Host_PrintArray(T* arr, size_t n, int precision = 4)
{
    cout << "[";
    for(size_t i = 0; i < n - 1; i++)
    {
        cout << std::setprecision(precision) << arr[i] << ", ";
    }
    cout << arr[n - 1] << "]" << endl;
}
 */
/**
 * \brief Helper function to print a host array easily.
 *
 * This function prints a host array to the screen using std::cout.
 * This will only work when running code on the host, and will only work
 * with a host array.
 *
 * NOTE: If calling this function gives an LNK1120 error, you must call it
 * with HelperFunctions::Host_PrintArray<[YOUR DESIRED TYPE]>(array, n). This
 * will tell the compiler to use [YOUR DESIRED TYPE] as the template type
 * @tparam T template type for generic functionality
 * @param arr host array of length n to be printed
 * @param n number of elements in host array
 * @param precision OPTIONAL: level of precision when printing values. Useful for control over float/double printing. Default = 4
 */
template<typename T>
[[maybe_unused]] __host__ void HelperFunctions::Host_PrintArray(T *arr, size_t n, int precision)
{
    cout << "[";
    for(size_t i = 0; i < n - 1; i++)
    {
        cout << std::setprecision(precision) << arr[i] << ", ";
    }
    cout << arr[n - 1] << "]" << endl;
}

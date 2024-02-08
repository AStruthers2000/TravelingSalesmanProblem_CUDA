//
// Created by Strut on 2/8/2024.
//

#ifndef VRP_HELPERFUNCTIONS_CUH
#define VRP_HELPERFUNCTIONS_CUH


class HelperFunctions
{
public:
    template <typename T>
    [[maybe_unused]] static __host__ void Host_PrintArray(T *arr, unsigned __int64 n, int precision = 4);
};




#endif //VRP_HELPERFUNCTIONS_CUH

#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>

#define CUDA_CHECK(f)                                                           \
                            if ((f) != cudaSuccess) {                           \
                                std::cerr << "CUDA MEM OP FAILED." << endl;     \
                                exit(EXIT_FAILURE);                             \
                            }                                                   \

__forceinline__ __device__ int GetThreadIdx() {
    // Return the index of the current thread.
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__forceinline__ __device__ double DeviceClamp(double val, double lo, double hi) {
    return val < lo ? lo : val > hi ? hi : val;
}

__forceinline__ __device__ void DecHex(int n, char* hex) {
    // Convert a decimal number to a (backwards) hex string.
    int i = 0;
    while (n != 0) {
        int temp = 0;  // Remainder.
        temp = n % 16;
        if (temp < 10) {
            // ASCII index of 0-9.
            hex[i] = temp + 48;
            i++;
        }
        else {
            // ASCII index of A-Z.
            hex[i] = temp + 55;
            i++;
        }
        n /= 16;
    }
}
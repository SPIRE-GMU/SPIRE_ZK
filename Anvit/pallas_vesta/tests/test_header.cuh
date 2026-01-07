#pragma once
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "./../include/Point.cuh"
#include "./../utils/field-helper.cuh"
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

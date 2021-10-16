#pragma once

#include <stdexcept>

#include "fmt/core.h"

#define CUDA_CHECK(expr) do { \
        cudaError_t res = expr; \
        if (res != cudaSuccess) { \
            throw std::runtime_error(fmt::format( \
                "CUDA call '{}' failed with {}({}) ({}:{})", \
                #expr, \
                cudaGetErrorName(res), \
                cudaGetErrorString(res), \
                __FILE__, \
                __LINE__ \
            )); \
        } \
    } while (0)

#define CU_CHECK(expr) do { \
        CUresult res = expr; \
        if (res != CUDA_SUCCESS) { \
            throw std::runtime_error(fmt::format( \
                "CUDA call '{}' failed with {} ({}:{})", \
                #expr, \
                res, \
                __FILE__, \
                __LINE__ \
            )); \
        } \
    } while (0)

#define OPTIX_CHECK(expr) do { \
        OptixResult res = expr; \
        if (res != OPTIX_SUCCESS) { \
            throw std::runtime_error(fmt::format( \
                "OptiX call '{}' failed with {} ({}:{})", \
                #expr, \
                res, \
                __FILE__, \
                __LINE__ \
            )); \
        } \
    } while (0)
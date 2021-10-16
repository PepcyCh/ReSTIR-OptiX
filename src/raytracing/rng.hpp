#pragma once

#include <cstdint>

#include "pcmath/cuda_macro_utils.hpp"

class RandomNumberGenerator {
public:
    CUDA_HOST_DEVICE void Seed(uint32_t seed) {
        state_ = seed;
    }

    CUDA_HOST_DEVICE float NextFloat(float l, float r) {
        const double rand01 = Next() / 4294967295.0;
        return rand01 * (r - l) + l;
    }

    CUDA_HOST_DEVICE int NextInt(int l, int r) {
        const double rand01 = Next() / 4294967296.0;
        return rand01 * (r - l) + l;
    }

private:
    CUDA_HOST_DEVICE uint32_t Next() {
        state_ ^= 2747636419u;
        state_ *= 2654435769u;
        state_ ^= state_ >> 16;
        state_ *= 2654435769u;
        state_ ^= state_ >> 16;
        state_ *= 2654435769u;
        return state_;
    }

    uint32_t state_;
};
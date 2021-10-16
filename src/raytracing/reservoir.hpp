#pragma once

#include "pcmath/pcmath.hpp"
#include "raytracing/rng.hpp"

struct ReservoirSample {
    pcm::Vec3 shade;
    float shade_lum; // p-hat
    pcm::Vec3 light_pos;
    pcm::Vec3 light_norm;
    pcm::Vec3 light_strength;
};

struct Reservoir {
    ReservoirSample out;
    float weight_sum;
    uint32_t num_samples;
    float w;

#ifdef __CUDACC__
    __device__
#endif
    static Reservoir New() {
        Reservoir res;
        memset(&res, 0, sizeof(Reservoir));
        return res;
    }

#ifdef __CUDACC__
    __device__
#endif
    void Clear() {
        memset(this, 0, sizeof(Reservoir));
    }

#ifdef __CUDACC__
    __device__
#endif
    void Update(
        const ReservoirSample &sample,
        float weight,
        uint32_t num_new_samples,
        RandomNumberGenerator &rng
    ) {
        weight_sum += weight;
        num_samples += num_new_samples;
        if (rng.NextFloat(0.0f, 1.0f) < weight / fmax(weight_sum, 0.001f)) {
            out = sample;
        }
    }

#ifdef __CUDACC__
    __device__
#endif
    void CalcW() {
        w = weight_sum / fmax(out.shade_lum * num_samples, 0.001f);
    }
};
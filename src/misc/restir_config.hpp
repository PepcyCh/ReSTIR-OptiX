#pragma once

#include <cstdint>

struct RestirConfig {
    uint8_t num_initial_samples;
    uint8_t num_eveluated_samples;
    uint8_t num_spatial_samples;
    uint8_t spatial_radius;
    uint8_t num_spatial_reuse_pass;
    bool temporal_reuse;
    bool visibility_reuse;
    bool unbiased;
    bool mis_spatial_reuse;
};
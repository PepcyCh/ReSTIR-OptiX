#pragma once

#include "pcmath/pcmath.hpp"

struct LightData {
    pcm::Mat4 model;
    pcm::Mat4 model_it;
    pcm::Vec3 strength;
    uint32_t index_offset;
    uint32_t vertex_offset;
    int at_another_index;
    float at_probability;
    float at_split;
};
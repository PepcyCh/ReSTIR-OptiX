#pragma once

#include "pcmath/pcmath.hpp"

struct CameraUniforms {
    pcm::Mat4 proj_view;
};

struct DrawableUniforms {
    pcm::Mat4 model;
    pcm::Mat4 model_it;
    uint32_t id;
};

struct MaterialUniforms {
    pcm::Vec4 base_color;
    pcm::Vec3 emissive;
    float metallic;
    float roughness;
    int alpha_mode;
    float alpha_cutoff;
    float normal_tex_scale;
};
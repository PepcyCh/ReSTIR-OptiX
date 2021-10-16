#pragma once

#include <cstdint>

#include "pcmath/pcmath.hpp"

struct Mesh {
    uint32_t first_index = 0;
    uint32_t num_indices = 0;
    uint32_t vertex_offset = 0;
    uint32_t num_vertices = 0;
    size_t material_index = -1;
    pcm::Vec3 bounding_min;
    pcm::Vec3 bounding_max;
};

struct Material {
    pcm::Vec4 base_color = pcm::Vec4(1.0f, 1.0f, 1.0f, 1.0f);
    int base_color_tex = -1;
    float metallic = 0.0f;
    float roughness = 1.0f;
    int metallic_roughness_tex = -1;
    pcm::Vec3 emissive = pcm::Vec3::Zero();
    int emissive_tex = -1;
    int alpha_mode = 0;
    float alpha_cutoff = 0.5f;
    bool double_sided = false;
    int normal_tex = -1;
    float normal_tex_scale = 1.0f;
    int occlusion_tex = -1;
    float occlusion_tex_strength = 1.0f;
};

struct DrawableNode {
    pcm::Mat4 model;
    size_t mesh_index = 0;
};
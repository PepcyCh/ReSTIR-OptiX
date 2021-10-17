#pragma once

#include "optix.h"

#include "misc/light_data.hpp"
#include "misc/restir_config.hpp"
#include "raytracing/reservoir.hpp"

struct LaunchParams {
    struct {
        pcm::Vec4 *albedo_emissive_buffer;
        pcm::Vec4 *pos_roughness_buffer;
        pcm::Vec4 *norm_metallic_buffer;
        uint32_t *id_buffer;
        uint32_t *prev_id_buffer;
        pcm::Vec4 *color_buffer;
        int curr_time;
        int width;
        int height;
    } frame;

    struct {
        pcm::Vec3 *positions;
        pcm::Vec3 *normals;
        uint32_t *indices;
        OptixTraversableHandle traversable;
    } scene;

    struct {
        pcm::Vec3 position;
        pcm::Mat4 proj_view;
        pcm::Mat4 prev_proj_view;
    } camera;

    struct {
        LightData *data;
        uint32_t light_count;
    } light;

    struct {
        RestirConfig config;
        Reservoir *prev_reservoirs;
        Reservoir *reservoirs;
    } restir;

    float light_strength_scale;
};

enum class RayType : uint32_t {
    eShadow,
    eCount
};
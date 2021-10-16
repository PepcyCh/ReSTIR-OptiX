#include "raytracing/launch_params.hpp"
#include "raytracing/raytracing_common.hpp"
#include "raytracing/reservoir.hpp"
#include "raytracing/shading.hpp"

__constant__ LaunchParams optix_launch_params;

struct RayPayload {
    bool visibility;
};

/*
static __device__ void SampleLight(
    const pcm::Vec3 &pos,
    RandomNumberGenerator &rng,
    pcm::Vec3 &light_dir,
    pcm::Vec3 &light_pos,
    pcm::Vec3 &light_norm,
    pcm::Vec3 &light_strength,
    float &light_dist,
    float &sample_pdf,
    float &light_attenuation
) {
    const auto &lights = optix_launch_params.light;
    const auto &scene = optix_launch_params.scene;

    float val = rng.NextFloat(0.0f, lights.light_count);
    uint32_t light_index = min(uint32_t(val), lights.light_count - 1);
    val -= light_index;
    if (val > lights.data[light_index].at_probability * lights.light_count) {
        light_index = lights.data[light_index].at_another_index;
    }

    const uint32_t vertex_offset = lights.data[light_index].vertex_offset;
    const uint32_t index_offset = lights.data[light_index].index_offset;
    const uint32_t i0 = scene.indices[index_offset];
    const uint32_t i1 = scene.indices[index_offset + 1];
    const uint32_t i2 = scene.indices[index_offset + 2];
    const pcm::Vec3 p0 = scene.positions[vertex_offset + i0];
    const pcm::Vec3 p1 = scene.positions[vertex_offset + i1];
    const pcm::Vec3 p2 = scene.positions[vertex_offset + i2];
    const pcm::Vec3 n0 = scene.normals[vertex_offset + i0];
    const pcm::Vec3 n1 = scene.normals[vertex_offset + i1];
    const pcm::Vec3 n2 = scene.normals[vertex_offset + i2];
    const pcm::Vec3 cross = (p1 - p0).Cross(p2 - p0);

    const float r0 = rng.NextFloat(0.0f, 1.0f);
    const float r0_sqrt = sqrt(r0);
    const float r1 = rng.NextFloat(0.0f, 1.0f);

    const float u = 1.0f - r0_sqrt;
    const float v = r0_sqrt * (1.0f - r1);
    const float w = 1.0f - u - v;
    light_pos = u * p0 + v * p1 + w * p2;
    const pcm::Vec3 norm = u * n0 + v * n1 + w * n2;

    const pcm::Vec3 light_vec = light_pos - pos;
    const float light_dist_sqr = light_vec.MagnitudeSqr();
    light_dist = sqrt(light_dist_sqr);
    light_dir = light_vec / light_dist;
    light_norm = norm;
    light_strength = lights.data[light_index].strength;
    light_attenuation = fmax(norm.Dot(-light_dir), 0.0f) / light_dist_sqr;

    sample_pdf = lights.data[light_index].at_probability / (cross.Length() * 0.5f * light_attenuation);
}
*/

OPTIX_CLOSESTHIT(Empty)() {}

OPTIX_ANYHIT(Empty)() {}

OPTIX_MISS(Shadow)() {
    RayPayload *payload = GetRayPayload<RayPayload>();
    payload->visibility = true;
}

OPTIX_RAYGEN(Lighting)() {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto &fr = optix_launch_params.frame;
    const auto &cam = optix_launch_params.camera;
    const auto &restir = optix_launch_params.restir;

    const uint32_t frame_index = ix + iy * fr.width;
    const uint32_t reservoir_index = frame_index * restir.config.num_eveluated_samples;

    RandomNumberGenerator rng;
    rng.Seed(frame_index + fr.curr_time);

    const pcm::Vec4 albedo_emissive = fr.albedo_emissive_buffer[frame_index];
    const pcm::Vec4 pos_roughness = fr.pos_roughness_buffer[frame_index];
    const pcm::Vec4 norm_metallic = fr.norm_metallic_buffer[frame_index];

    const pcm::Vec3 pos = pcm::Vec3(pos_roughness);
    const pcm::Vec3 norm = pcm::Vec3(norm_metallic);
    const pcm::Vec3 base_color = pcm::Vec3(albedo_emissive);
    const float roughness = pos_roughness.W();
    const float metallic = norm_metallic.W();
    const bool is_emissive = albedo_emissive.W() > 0.0f;

    const pcm::Vec3 view_dir = (cam.position - pos).Normalize();

    if (norm.MagnitudeSqr() > 0.8f) {
        if (is_emissive) {
            fr.color_buffer[frame_index] = pcm::Vec4(base_color, 1.0f);
        } else {
            pcm::Vec3 color = pcm::Vec3::Zero();
            for (uint8_t i = 0; i < restir.config.num_eveluated_samples; i++) {
                const Reservoir &reservoir = restir.reservoirs[reservoir_index + i];
                color += reservoir.out.shade * reservoir.w;
            }
            color /= restir.config.num_eveluated_samples;
            fr.color_buffer[frame_index] = pcm::Vec4(color, 1.0f);
            // TODO - debug
            // if (isnan(color.X()) || isnan(color.Y()) || isnan(color.Z())) {
            //     fr.color_buffer[frame_index] = pcm::Vec4(0.0f, 0.0f, 1.0f, 1.0f);
            // }
        }
    } else {
        fr.color_buffer[frame_index] = pcm::Vec4::Zero();
    }
}
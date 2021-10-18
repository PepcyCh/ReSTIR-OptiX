#include "raytracing/launch_params.hpp"
#include "raytracing/raytracing_common.hpp"
#include "raytracing/reservoir.hpp"
#include "raytracing/shading.hpp"

__constant__ LaunchParams optix_launch_params;

struct RayPayload {
    bool visibility;
};

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
    const uint32_t curr_id = fr.id_buffer[frame_index];

    const pcm::Vec3 pos = pcm::Vec3(pos_roughness);
    const pcm::Vec3 norm = pcm::Vec3(norm_metallic);
    const pcm::Vec3 base_color = pcm::Vec3(albedo_emissive);
    const float roughness = pos_roughness.W();
    const float metallic = norm_metallic.W();
    const bool is_emissive = albedo_emissive.W() > 0.0f;

    const pcm::Vec3 view_dir = (cam.position - pos).Normalize();

    if (curr_id != 0) {
        if (is_emissive) {
            fr.color_buffer[frame_index] = pcm::Vec4(base_color, 1.0f);
        } else {
            pcm::Vec3 color = pcm::Vec3::Zero();
            for (uint8_t i = 0; i < restir.config.num_eveluated_samples; i++) {
                const Reservoir &reservoir = restir.reservoirs[reservoir_index + i];

                if (!restir.config.visibility_reuse
                    || (restir.config.num_spatial_reuse_pass > 0 && !restir.config.unbiased)) {
                    RayPayload shadow_payload;
                    shadow_payload.visibility = false;

                    const pcm::Vec3 light_vec = reservoir.out.light_pos - pos;
                    const float light_dist = light_vec.Length();
                    const pcm::Vec3 light_dir = light_vec / light_dist;

                    RayDesc ray;
                    ray.origin = pos;
                    ray.direction = light_dir;
                    ray.t_min = 0.001f;
                    ray.t_max = light_dist - 0.001f;

                    TraceRay(
                        optix_launch_params.scene.traversable,
                        OPTIX_RAY_FLAG_DISABLE_ANYHIT
                            | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
                            | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                        0xff,
                        0,
                        1,
                        0,
                        ray,
                        &shadow_payload
                    );

                    if (shadow_payload.visibility) {
                        color += reservoir.out.shade * reservoir.w;
                    }
                } else {
                    color += reservoir.out.shade * reservoir.w;
                }
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
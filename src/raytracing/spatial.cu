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

OPTIX_RAYGEN(Spatial)() {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto &fr = optix_launch_params.frame;
    const auto &cam = optix_launch_params.camera;
    const auto &restir = optix_launch_params.restir;

    const uint32_t frame_index = ix + iy * fr.width;
    const uint32_t reservoir_index = frame_index * restir.config.num_eveluated_samples;

    const pcm::Vec4 albedo_emissive = fr.albedo_emissive_buffer[frame_index];
    const pcm::Vec4 pos_roughness = fr.pos_roughness_buffer[frame_index];
    const pcm::Vec4 norm_metallic = fr.norm_metallic_buffer[frame_index];
    const uint32_t curr_id = fr.id_buffer[frame_index];

    const pcm::Vec3 pos = pcm::Vec3(pos_roughness);
    const pcm::Vec3 norm = pcm::Vec3(norm_metallic);
    const pcm::Vec3 base_color = pcm::Vec3(albedo_emissive);
    const float roughness = pos_roughness.W();
    const float metallic = norm_metallic.W();
    
    const pcm::Vec3 view_vec = cam.position - pos;
    const float camera_dist = view_vec.Length();
    const pcm::Vec3 view_dir = view_vec / camera_dist;

    RandomNumberGenerator rng;
    rng.Seed(frame_index + fr.curr_time);

    if (curr_id != 0) {
        for (uint8_t i = 0; i < restir.config.num_eveluated_samples; i++) {
            Reservoir reservoir = restir.prev_reservoirs[reservoir_index + i];

            uint32_t spatial_samples[8];
            uint8_t valid_spatial_samples = 0;
            uint32_t initial_num_samples = reservoir.num_samples;
            // float samples_weight[8];
            // float initial_weight = reservoir.weight_sum;

            for (uint8_t j = 0; j < restir.config.num_spatial_samples; j++) {
                const int dx = rng.NextInt(-restir.config.spatial_radius, restir.config.spatial_radius + 1);
                const int dy = rng.NextInt(-restir.config.spatial_radius, restir.config.spatial_radius + 1);
                const int x = max(min(ix + dx, fr.width - 1), 0);
                const int y = max(min(iy + dy, fr.height - 1), 0);
                const uint32_t neighbor_frame_index = x + y * fr.width;
                const uint32_t neighbor_reservoir_index = neighbor_frame_index * restir.config.num_eveluated_samples;

                const pcm::Vec3 neighbor_pos = pcm::Vec3(fr.pos_roughness_buffer[neighbor_frame_index]);
                const pcm::Vec3 neighbor_norm = pcm::Vec3(fr.norm_metallic_buffer[neighbor_frame_index]);

                const pcm::Vec3 neighbor_view_vec = cam.position - neighbor_pos;
                const float neighbor_camera_dist = neighbor_view_vec.Length();
                const pcm::Vec3 neighbor_view_dir = neighbor_view_vec / neighbor_camera_dist;

                const float camera_dist_ratio = fabs(neighbor_camera_dist - camera_dist) / camera_dist;
                const float norm_diff = norm.Dot(neighbor_norm);
                if (camera_dist_ratio > 0.1f || norm_diff < 0.9063077870366499f) { // cos(25 deg)
                    continue;
                }

                Reservoir neighbor = restir.prev_reservoirs[neighbor_reservoir_index + i];

                const pcm::Vec3 light_vec = neighbor.out.light_pos - pos;
                const float light_dist_sqr = light_vec.MagnitudeSqr();
                const float light_dist = sqrt(light_dist_sqr);
                const pcm::Vec3 light_dir = light_vec / light_dist;

                if (restir.config.unbiased && restir.config.visibility_reuse) {
                    RayPayload shadow_payload;
                    shadow_payload.visibility = false;
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
                    if (!shadow_payload.visibility) {
                        neighbor.w = 0.0f;
                    }
                }

                const float atten = max(neighbor.out.light_norm.Dot(-light_dir), 0.0f) / max(light_dist_sqr, 0.001f);
                neighbor.out.shade = Shade(
                    view_dir,
                    light_dir,
                    norm,
                    neighbor.out.light_strength * atten,
                    base_color,
                    roughness,
                    metallic
                );
                neighbor.out.shade_lum = Luminance(neighbor.out.shade);
                const float weight = neighbor.out.shade_lum * neighbor.w * neighbor.num_samples;
                reservoir.Update(neighbor.out, weight, neighbor.num_samples, rng);

                spatial_samples[valid_spatial_samples] = neighbor_frame_index;
                // samples_weight[valid_spatial_samples] = weight;
                ++valid_spatial_samples;
            }

            if (restir.config.unbiased) {
                reservoir.num_samples = initial_num_samples;
                // reservoir.weight_sum = initial_weight;
                for (uint8_t j = 0; j < valid_spatial_samples; j++) {
                    const pcm::Vec3 neighbor_pos = pcm::Vec3(fr.pos_roughness_buffer[spatial_samples[j]]);
                    const pcm::Vec3 neighbor_norm = pcm::Vec3(fr.norm_metallic_buffer[spatial_samples[j]]);

                    const pcm::Vec3 neighbor_light_vec = reservoir.out.light_pos - neighbor_pos;
                    const float neighbor_light_dist = neighbor_light_vec.Length();
                    const pcm::Vec3 neighbor_light = neighbor_light_vec / neighbor_light_dist;

                    bool shadowed = false;
                    if (restir.config.visibility_reuse) {
                        RayPayload shadow_payload;
                        shadow_payload.visibility = false;

                        RayDesc ray;
                        ray.origin = neighbor_pos;
                        ray.direction = neighbor_light;
                        ray.t_min = 0.001f;
                        ray.t_max = neighbor_light_dist - 0.001f;

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
                            // reservoir.weight_sum += samples_weight[j];
                        } else {
                            shadowed = true;
                        }
                    } else {
                        // reservoir.weight_sum += samples_weight[j];
                    }

                    if (shadowed || neighbor_light.Dot(neighbor_norm) <= 0.0f
                        || neighbor_light.Dot(reservoir.out.light_norm) >= 0.0f) {
                        continue;
                    }

                    reservoir.num_samples += restir.prev_reservoirs[spatial_samples[j]
                        * restir.config.num_eveluated_samples + i].num_samples;
                }
            }
            
            reservoir.CalcW();
            restir.reservoirs[reservoir_index + i] = reservoir;
        }
    }
}
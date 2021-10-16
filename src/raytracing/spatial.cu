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

    if (norm.MagnitudeSqr() > 0.8f) {
        for (uint8_t i = 0; i < restir.config.num_eveluated_samples; i++) {
            Reservoir reservoir = restir.prev_reservoirs[reservoir_index + i];

            for (uint8_t j = 0; j < restir.config.num_spatial_samples; j++) {
                const int dx = rng.NextInt(-restir.config.spatial_radius, restir.config.spatial_radius + 1);
                const int dy = rng.NextInt(-restir.config.spatial_radius, restir.config.spatial_radius + 1);
                const int x = max(min(ix + dx, fr.width - 1), 0);
                const int y = max(min(iy + dy, fr.height - 1), 0);
                const int neighbor_frame_index = x + y * fr.width;
                const int neighbor_reservoir_index = neighbor_frame_index * restir.config.num_eveluated_samples;

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
                const float light_attenuation =
                    max(neighbor.out.light_norm.Dot(-light_dir), 0.0f) / light_dist_sqr;
                const pcm::Vec3 attenuated_light_strength = neighbor.out.light_strength * light_attenuation;
                neighbor.out.shade = Shade(
                    view_dir,
                    light_dir,
                    norm,
                    attenuated_light_strength,
                    base_color,
                    roughness,
                    metallic
                );
                neighbor.out.shade_lum = Luminance(neighbor.out.shade);
                const float weight = neighbor.out.shade_lum * neighbor.w * neighbor.num_samples;
                reservoir.Update(neighbor.out, weight, neighbor.num_samples, rng);
            }

            reservoir.CalcW();
            restir.reservoirs[reservoir_index + i] = reservoir;
        }
    } else {
        for (uint8_t i = 0; i < restir.config.num_eveluated_samples; i++) {
            restir.reservoirs[reservoir_index + i].w = 0.0f;
            restir.reservoirs[reservoir_index + i].num_samples = 0;
        }
    }
}
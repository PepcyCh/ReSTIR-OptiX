#include "raytracing/launch_params.hpp"
#include "raytracing/raytracing_common.hpp"
#include "raytracing/reservoir.hpp"
#include "raytracing/shading.hpp"

__constant__ LaunchParams optix_launch_params;

struct RayPayload {
    bool visibility;
};

static __device__ void SampleLight(
    const pcm::Vec3 &pos,
    RandomNumberGenerator &rng,
    pcm::Vec3 &light_dir,
    pcm::Vec3 &light_pos,
    pcm::Vec3 &light_norm,
    pcm::Vec3 &light_strength,
    float &light_dist,
    float &sample_pdf
) {
    const auto &lights = optix_launch_params.light;
    const auto &scene = optix_launch_params.scene;

    float val = rng.NextFloat(0.0f, lights.light_count);
    uint32_t light_index = min(uint32_t(val), lights.light_count - 1);
    val -= light_index;
    if (val > lights.data[light_index].at_probability * lights.light_count) {
        light_index = lights.data[light_index].at_another_index;
    }

    const pcm::Mat4 &model = lights.data[light_index].model;
    const pcm::Mat4 &model_it = lights.data[light_index].model_it;
    const uint32_t vertex_offset = lights.data[light_index].vertex_offset;
    const uint32_t index_offset = lights.data[light_index].index_offset;

    const uint32_t i0 = scene.indices[index_offset];
    const uint32_t i1 = scene.indices[index_offset + 1];
    const uint32_t i2 = scene.indices[index_offset + 2];
    const pcm::Vec3 p0 = pcm::Vec3(model * pcm::Vec4(scene.positions[vertex_offset + i0], 1.0f));
    const pcm::Vec3 p1 = pcm::Vec3(model * pcm::Vec4(scene.positions[vertex_offset + i1], 1.0f));
    const pcm::Vec3 p2 = pcm::Vec3(model * pcm::Vec4(scene.positions[vertex_offset + i2], 1.0f));
    const pcm::Vec3 n0 = (pcm::Mat3(model_it) * scene.normals[vertex_offset + i0]).Normalize();
    const pcm::Vec3 n1 = (pcm::Mat3(model_it) * scene.normals[vertex_offset + i1]).Normalize();
    const pcm::Vec3 n2 = (pcm::Mat3(model_it) * scene.normals[vertex_offset + i2]).Normalize();
    const pcm::Vec3 cross = (p1 - p0).Cross(p2 - p0);

    const float r0 = rng.NextFloat(0.0f, 1.0f);
    const float r0_sqrt = sqrt(r0);
    const float r1 = rng.NextFloat(0.0f, 1.0f);

    const float u = 1.0f - r0_sqrt;
    const float v = r0_sqrt * (1.0f - r1);
    const float w = 1.0f - u - v;
    light_pos = u * p0 + v * p1 + w * p2;
    const pcm::Vec3 norm = (u * n0 + v * n1 + w * n2).Normalize();

    const pcm::Vec3 light_vec = light_pos - pos;
    const float light_dist_sqr = light_vec.MagnitudeSqr();
    light_dist = sqrt(light_dist_sqr);
    light_dir = light_vec / light_dist;
    light_norm = norm;
    light_strength = lights.data[light_index].strength;

    sample_pdf = lights.data[light_index].at_probability * light_dist_sqr
        / max(cross.Length() * 0.5f * max(norm.Dot(-light_dir), 0.0f), 0.001f);
}

OPTIX_CLOSESTHIT(Empty)() {}

OPTIX_ANYHIT(Empty)() {}

OPTIX_MISS(Shadow)() {
    RayPayload *payload = GetRayPayload<RayPayload>();
    payload->visibility = true;
}

OPTIX_RAYGEN(Sample)() {
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

    const pcm::Vec3 view_dir = (cam.position - pos).Normalize();

    if (curr_id != 0) {
        int prev_reservoir_index = -1;
        if (restir.config.temporal_reuse && restir.prev_reservoirs) {
            const pcm::Vec4 prev_uv_homo = cam.prev_proj_view * pcm::Vec4(pos, 1.0f);
            const pcm::Vec3 prev_uv_ndc = pcm::Vec3(prev_uv_homo) / prev_uv_homo.W();
            if (prev_uv_ndc.Z() >= 0.0f && prev_uv_ndc.Z() <= 1.0f) {
                const int prev_x = (prev_uv_ndc.X() + 1.0f) * 0.5f * fr.width;
                const int prev_y = (prev_uv_ndc.Y() + 1.0f) * 0.5f * fr.height;
                if (prev_x >= 0 && prev_x < fr.width && prev_y >= 0 && prev_y < fr.height) {
                    const int prev_frame_index = prev_x + prev_y * fr.width;
                    const uint32_t prev_id = fr.prev_id_buffer[prev_frame_index];
                    if (curr_id == prev_id) {
                        prev_reservoir_index = prev_frame_index * restir.config.num_eveluated_samples;
                    }
                }
            }
        }

        for (uint8_t i = 0; i < restir.config.num_eveluated_samples; i++) {
            Reservoir reservoir = Reservoir::New();

            for (uint8_t j = 0; j < restir.config.num_initial_samples; j++) {
                ReservoirSample sample;

                pcm::Vec3 light_dir;
                float light_dist;
                float light_pdf;
                SampleLight(
                    pos,
                    rng,
                    light_dir,
                    sample.light_pos,
                    sample.light_norm,
                    sample.light_strength,
                    light_dist,
                    light_pdf
                );

                sample.shade = Shade(
                    view_dir,
                    light_dir,
                    norm,
                    sample.light_strength,
                    base_color,
                    roughness,
                    metallic
                );
                sample.shade_lum = Luminance(sample.shade);

                reservoir.Update(sample, sample.shade_lum / max(light_pdf, 0.001f), 1, rng);
            }

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

            if (!shadow_payload.visibility) {
                reservoir.Clear();
            } else if (prev_reservoir_index >= 0) {
                Reservoir prev_reservoir = restir.prev_reservoirs[prev_reservoir_index + i];
                const uint32_t clamped_num_samples = min(prev_reservoir.num_samples, 20 * reservoir.num_samples);
                const float weight = prev_reservoir.out.shade_lum * prev_reservoir.w * clamped_num_samples;
                reservoir.Update(prev_reservoir.out, weight, clamped_num_samples, rng);

                reservoir.CalcW();
            } else {
                reservoir.CalcW();
            }

            restir.reservoirs[reservoir_index + i] = reservoir;
        }
    }
}
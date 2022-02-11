#include "misc/lights.hpp"

#include <algorithm>

namespace ranges = std::ranges;

Lights::Lights(const Scene *scene) {
    float sum = 0.0f;

    for (const auto &drawable : scene->Drawables()) {
        const auto &mesh = scene->Meshes()[drawable.mesh_index];
        const auto &material = scene->Materials()[mesh.material_index];
        if (material.emissive.MagnitudeSqr() > 0.0) {
            for (uint32_t i = 0; i < mesh.num_indices; i += 3) {
                const uint32_t i0 = scene->Indices()[mesh.first_index + i];
                const uint32_t i1 = scene->Indices()[mesh.first_index + i + 1];
                const uint32_t i2 = scene->Indices()[mesh.first_index + i + 2];
                const pcm::Vec3 p0(drawable.model * pcm::Vec4(scene->Positions()[mesh.vertex_offset + i0]));
                const pcm::Vec3 p1(drawable.model * pcm::Vec4(scene->Positions()[mesh.vertex_offset + i1]));
                const pcm::Vec3 p2(drawable.model * pcm::Vec4(scene->Positions()[mesh.vertex_offset + i2]));
                const float area = (p1 - p0).Cross(p2 - p0).Length();
                const float luminance =
                    0.299f * material.emissive.X() + 0.597f * material.emissive.Y() + 0.114f * material.emissive.Z();

                LightData data {
                    .model = drawable.model,
                    .model_it = drawable.model.Inverse().Transpose(),
                    .strength = material.emissive,
                    .index_offset = mesh.first_index + i,
                    .vertex_offset = mesh.vertex_offset,
                    .at_another_index = -1,
                    .at_probability = area * luminance,
                };
                sum += data.at_probability;
                lights_.emplace_back(data);
            }
        }
    }

    const float sum_inv = 1.0f / sum;
    for (LightData &data : lights_) {
        data.at_probability *= sum_inv;
    }

    std::vector<float> u(lights_.size());
    ranges::transform(lights_, u.begin(), [&](const LightData &data) { return data.at_probability * u.size(); });
    size_t poor = ranges::find_if(u, [](float u) { return u < 1.0f; }) - u.begin();
    size_t poor_max = poor;
    size_t rich = ranges::find_if(u, [](float u) { return u > 1.0f; }) - u.begin();
    while (poor < u.size() && rich < u.size()) {
        const float diff = 1.0f - u[poor];
        u[rich] -= diff;
        lights_[poor].at_another_index = rich;

        if (u[rich] < 1.0 && rich < poor_max) {
            poor = rich;
        } else {
            poor = ranges::find_if(u.begin() + poor_max + 1, u.end(), [](float u) { return u < 1.0f; }) - u.begin();
            poor_max = poor;
        }

        rich = ranges::find_if(u.begin() + rich, u.end(), [](float u) { return u > 1.0f; }) - u.begin();
    }

    for (size_t i = 0; i < lights_.size(); i++) {
        lights_[i].at_split = u[i];
    }
}
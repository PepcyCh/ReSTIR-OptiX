#pragma once

#include <algorithm>

#include "pcmath/pcmath.hpp"

static __device__ float Luminance(const pcm::Vec3 &color) {
    return 0.299f * color.X() + 0.597f * color.Y() + 0.114f * color.Z();
}

static __device__ float Pow2(float x) {
    return x * x;
}

static __device__ float Pow5(float x) {
    return x * x * x * x * x;
}

static __device__ pcm::Vec3 SchlickFresnel(const pcm::Vec3 &r0, float cos) {
    return r0 + (pcm::Vec3(1.0f, 1.0f, 1.0f) - r0) * Pow5(1.0f - cos);
}

static __device__ float SchlickFresnel(float r0, float cos) {
    return r0 + (1.0f - r0) * Pow5(1.0f - cos);
}

static __device__ float NdfGgx(float ndoth, float a2) {
    return a2 / max(3.14159265359f * Pow2(ndoth * ndoth * (a2 - 1.0f) + 1.0f), 0.001f);
}

static __device__ float SeparableVisible(float ndotv, float ndotl, float a2) {
    float v = fabs(ndotv) + sqrt((1.0f - a2) * ndotv * ndotv + a2);
    float l = fabs(ndotl) + sqrt((1.0f - a2) * ndotl * ndotl + a2);
    return 1.0 / max(v * l, 0.001f);
}

static __device__ pcm::Vec3 Shade(
    const pcm::Vec3 &view_dir,
    const pcm::Vec3 &light_dir,
    const pcm::Vec3 &normal,
    const pcm::Vec3 &light_strength,
    const pcm::Vec3 &base_color,
    float roughness,
    float metallic
) {
    const pcm::Vec3 half_dir = (view_dir + light_dir).Normalize();
    const float a = roughness * roughness;
    const float a2 = a * a;

    const float ndoth = max(normal.Dot(half_dir), 0.0f);
    const float ndotv = max(normal.Dot(view_dir), 0.0f);
    const float ndotl = max(normal.Dot(light_dir), 0.0f);
    const float hdotv = max(view_dir.Dot(half_dir), 0.0f);

    const pcm::Vec3 diffuse = base_color / 3.14159265359f;

    const float ndf = NdfGgx(ndoth, a2);
    const float visible = SeparableVisible(ndotv, ndotl, a2);

    const pcm::Vec3 dielectric_fresnel = SchlickFresnel(pcm::Vec3(0.04f, 0.04f, 0.04f), hdotv);
    const pcm::Vec3 metal_fresnel = SchlickFresnel(base_color, hdotv);

    const pcm::Vec3 dielectric_res = dielectric_fresnel * ndf * visible
        + (pcm::Vec3(1.0f, 1.0f, 1.0f) - dielectric_fresnel) * diffuse;
    const pcm::Vec3 metal_res = metal_fresnel * ndf * visible;
    const pcm::Vec3 res = metallic * metal_res + (1.0f - metallic) * dielectric_res;

    return res * light_strength * ndotl;
}
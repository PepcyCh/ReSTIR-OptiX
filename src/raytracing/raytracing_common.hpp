#pragma once

#include "optix_device.h"

#include "pcmath/pcmath.hpp"

#define OPTIX_RAYGEN(name) extern "C" __global__ void __raygen__##name
#define OPTIX_MISS(name) extern "C" __global__ void __miss__##name
#define OPTIX_CLOSESTHIT(name) extern "C" __global__ void __closesthit__##name
#define OPTIX_ANYHIT(name) extern "C" __global__ void __anyhit__##name

struct RayDesc {
    pcm::Vec3 origin;
    float t_min;
    pcm::Vec3 direction;
    float t_max;
    float time;
};

static __forceinline__ __device__ void TraceRay(
    OptixTraversableHandle traversable,
    uint32_t ray_flags,
    uint32_t visibility_mask,
    uint32_t sbt_offset,
    uint32_t sbt_stride,
    uint32_t miss_sbt_index,
    RayDesc ray,
    void *payload
) {
    const float3 pos = make_float3(ray.origin.X(), ray.origin.Y(), ray.origin.Z());
    const float3 dir = make_float3(ray.direction.X(), ray.direction.Y(), ray.direction.Z());

    const uint64_t ptr_addr = reinterpret_cast<uint64_t>(payload);
    uint32_t u0 = ptr_addr >> 32;
    uint32_t u1 = ptr_addr & 0x00000000'ffffffff;

    optixTrace(
        traversable,
        pos,
        dir,
        ray.t_min,
        ray.t_max,
        ray.time,
        OptixVisibilityMask(visibility_mask),
        ray_flags,
        sbt_offset,
        sbt_stride,
        miss_sbt_index,
        u0,
        u1
    );
}

template<typename Payload>
static __forceinline__ __device__ Payload *GetRayPayload() {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    const uint64_t ptr_addr = (static_cast<uint64_t>(u0) << 32) | u1;
    return reinterpret_cast<Payload *>(ptr_addr);
}

static __forceinline__ __device__ pcm::Vec3 GetWorldRayDirection() {
    float3 ray_dir = optixGetWorldRayDirection();
    return pcm::Vec3(ray_dir.x, ray_dir.y, ray_dir.z);
}
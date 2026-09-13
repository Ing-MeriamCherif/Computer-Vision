#include <optix.h>

struct LaunchParams {
    OptixTraversableHandle traversable;
    unsigned long long points;
    unsigned long long normals;
    unsigned long long visibility;
    unsigned int width;
    unsigned int height;
    unsigned int light_count;
    float light_positions[6];
};

extern "C" {
__constant__ LaunchParams params;
}

static __forceinline__ __device__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static __forceinline__ __device__ float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static __forceinline__ __device__ float3 scale3(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
static __forceinline__ __device__ float length3(float3 a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

extern "C" __global__ void __raygen__shadow() {
    const uint3 index = optixGetLaunchIndex();
    const unsigned int pixel = index.y * params.width + index.x;
    const unsigned int light_index = index.z;
    float* output = reinterpret_cast<float*>(params.visibility);
    if (light_index >= params.light_count) {
        output[light_index * params.width * params.height + pixel] = 1.0f;
        return;
    }

    const float3* points = reinterpret_cast<const float3*>(params.points);
    const float3* normals = reinterpret_cast<const float3*>(params.normals);
    const float3 point = points[pixel];
    const float3 normal = normals[pixel];
    if (point.z <= 1.0e-5f || !isfinite(point.x) || !isfinite(point.y) || !isfinite(point.z)) {
        output[light_index * params.width * params.height + pixel] = 1.0f;
        return;
    }

    const float3 light = make_float3(
        params.light_positions[light_index * 3],
        params.light_positions[light_index * 3 + 1],
        params.light_positions[light_index * 3 + 2]);
    const float3 delta = sub3(light, point);
    const float distance = length3(delta);
    if (distance <= 0.003f) {
        output[light_index * params.width * params.height + pixel] = 1.0f;
        return;
    }

    const float3 origin = add3(point, scale3(normal, 0.002f));
    const float3 direction = scale3(delta, 1.0f / distance);
    unsigned int visible = 0;
    optixTrace(
        params.traversable,
        origin,
        direction,
        0.001f,
        max(distance - 0.004f, 0.001f),
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,
        1,
        0,
        visible);
    output[light_index * params.width * params.height + pixel] = visible ? 1.0f : 0.18f;
}

extern "C" __global__ void __miss__shadow() {
    optixSetPayload_0(1);
}

extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(0);
}

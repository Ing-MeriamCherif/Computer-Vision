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
    float source_radii[2];
    float self_eps[2];
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
static __forceinline__ __device__ float3 cross3(float3 a, float3 b) {
    return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
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

    const float3 origin = add3(point, scale3(normal, params.self_eps[light_index]));
    const float3 direction = scale3(delta, 1.0f / distance);
    float3 reference = fabsf(direction.z) < 0.9f ? make_float3(0,0,1) : make_float3(0,1,0);
    float3 tangent = scale3(cross3(direction, reference), 1.0f / max(length3(cross3(direction, reference)), 1e-5f));
    float3 bitangent = cross3(direction, tangent);
    const float samples[4][2] = {{0.0f,0.0f},{0.65f,0.0f},{-0.325f,0.563f},{-0.325f,-0.563f}};
    unsigned int visible_count = 0;
    for (int sample=0; sample<4; ++sample) {
        float3 emitter = add3(light, scale3(add3(scale3(tangent, samples[sample][0]), scale3(bitangent, samples[sample][1])), params.source_radii[light_index]));
        float3 to_emitter = sub3(emitter, point); float d = length3(to_emitter);
        unsigned int visible = 0;
        optixTrace(params.traversable, origin, scale3(to_emitter, 1.0f/max(d,1e-5f)), 0.001f,
            max(d - params.self_eps[light_index], 0.001f), 0.0f, OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, visible);
        visible_count += visible ? 1u : 0u;
    }
    output[light_index * params.width * params.height + pixel] = float(visible_count) / 4.0f;
}

extern "C" __global__ void __miss__shadow() {
    optixSetPayload_0(1);
}

extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(0);
}

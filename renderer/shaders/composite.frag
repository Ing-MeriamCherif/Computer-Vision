#version 330

uniform sampler2D u_ambient;
uniform sampler2D u_direct;
uniform sampler2D u_shadow_visibility;
uniform sampler2D u_depth;
uniform sampler2D u_depth_valid;
uniform int u_shadow_softening_enabled;
uniform int u_shadow_soft_samples;
uniform int u_shadow_edge_aware_upsampling;
uniform float u_shadow_soft_radius;
uniform float u_depth_edge_threshold_m;

in vec2 v_uv;
out vec4 frag_color;

const ivec2 BILINEAR_OFFSETS[4] = ivec2[4](
    ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 1)
);
const vec2 SOFT_OFFSETS[8] = vec2[8](
    vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0),
    vec2(0.70710678, 0.70710678), vec2(-0.70710678, -0.70710678),
    vec2(0.70710678, -0.70710678), vec2(-0.70710678, 0.70710678)
);

vec3 linear_to_srgb(vec3 linear_rgb) {
    vec3 clamped = clamp(linear_rgb, 0.0, 1.0);
    vec3 low = clamped * 12.92;
    vec3 high = 1.055 * pow(clamped, vec3(1.0 / 2.4)) - 0.055;
    return mix(low, high, step(vec3(0.0031308), clamped));
}

float depth_edge_weight(ivec2 shadow_coord, float receiver_z, bool receiver_valid) {
    if (u_shadow_edge_aware_upsampling == 0 || !receiver_valid) {
        return 1.0;
    }
    ivec2 shadow_size = textureSize(u_shadow_visibility, 0);
    // The shadow pass is a framebuffer texture (bottom-left origin), while
    // uploaded camera depth uses top-left image coordinates.
    vec2 tap_uv = vec2(
        (float(shadow_coord.x) + 0.5) / float(shadow_size.x),
        1.0 - (float(shadow_coord.y) + 0.5) / float(shadow_size.y)
    );
    float tap_z = texture(u_depth, tap_uv).r;
    float tap_valid = texture(u_depth_valid, tap_uv).r;
    if (tap_valid <= 0.0 || isnan(tap_z) || isinf(tap_z) || tap_z <= 0.0) {
        return 0.0;
    }
    float difference_m = abs(receiver_z - tap_z);
    return max(0.0, 1.0 - difference_m / u_depth_edge_threshold_m);
}

float bilinear_visibility(vec2 shadow_position, float receiver_z, bool receiver_valid) {
    ivec2 shadow_size = textureSize(u_shadow_visibility, 0);
    if (u_shadow_edge_aware_upsampling == 0) {
        // P6 fallback: ordinary bilinear hardware upsampling.
        vec2 shadow_uv = (shadow_position + vec2(0.5)) / vec2(shadow_size);
        return texture(u_shadow_visibility, shadow_uv).r;
    }

    ivec2 base = ivec2(floor(shadow_position));
    vec2 fraction = fract(shadow_position);
    float visibility_sum = 0.0;
    float weight_sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        ivec2 raw_coord = base + BILINEAR_OFFSETS[i];
        ivec2 coord = clamp(raw_coord, ivec2(0), shadow_size - ivec2(1));
        vec2 side = vec2(BILINEAR_OFFSETS[i]);
        vec2 axis_weight = mix(vec2(1.0) - fraction, fraction, side);
        float spatial_weight = axis_weight.x * axis_weight.y;
        float edge_weight = depth_edge_weight(coord, receiver_z, receiver_valid);
        float weight = spatial_weight * edge_weight;
        visibility_sum += texelFetch(u_shadow_visibility, coord, 0).r * weight;
        weight_sum += weight;
    }

    // No reliable same-surface taps means unknown, so stay unshadowed.
    if (weight_sum <= 1e-6) {
        return 1.0;
    }
    return visibility_sum / weight_sum;
}

float filtered_visibility(float receiver_z, bool receiver_valid) {
    ivec2 shadow_size = textureSize(u_shadow_visibility, 0);
    // Convert top-left camera UV into the bottom-left shadow FBO texture space.
    vec2 shadow_uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec2 shadow_position = shadow_uv * vec2(shadow_size) - vec2(0.5);
    float center = bilinear_visibility(shadow_position, receiver_z, receiver_valid);
    if (u_shadow_softening_enabled == 0) {
        return center;
    }

    float visibility_sum = center;
    float weight_sum = 1.0;
    for (int i = 0; i < 8; ++i) {
        if (i >= u_shadow_soft_samples) {
            break;
        }
        vec2 offset_position = shadow_position + SOFT_OFFSETS[i] * u_shadow_soft_radius;
        float tap = bilinear_visibility(offset_position, receiver_z, receiver_valid);
        visibility_sum += tap;
        weight_sum += 1.0;
    }
    return visibility_sum / weight_sum;
}

void main() {
    // Lighting and shadow outputs were rasterized into bottom-left-origin FBOs.
    vec2 layer_uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec3 ambient = texture(u_ambient, layer_uv).rgb;
    vec3 direct = texture(u_direct, layer_uv).rgb;
    float receiver_z = texture(u_depth, v_uv).r;
    float receiver_valid_value = texture(u_depth_valid, v_uv).r;
    bool receiver_valid = receiver_valid_value > 0.0
        && !isnan(receiver_z) && !isinf(receiver_z) && receiver_z > 0.0;
    float visibility = clamp(filtered_visibility(receiver_z, receiver_valid), 0.0, 1.0);
    vec3 result_linear = ambient + visibility * direct;
    if (any(isnan(result_linear)) || any(isinf(result_linear))) {
        result_linear = ambient;
    }
    frag_color = vec4(linear_to_srgb(result_linear), 1.0);
}

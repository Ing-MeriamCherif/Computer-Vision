#version 330

uniform sampler2D u_ambient;
uniform sampler2D u_direct_1;
uniform sampler2D u_direct_2;
uniform sampler2D u_shadow_visibility_1;
uniform sampler2D u_shadow_visibility_2;
uniform sampler2D u_depth;
uniform sampler2D u_depth_valid;
uniform sampler2D u_volumetric;
uniform int u_volumetric_enabled;
uniform int u_shadow_softening_enabled_1;
uniform int u_shadow_softening_enabled_2;
uniform int u_shadow_soft_samples_1;
uniform int u_shadow_soft_samples_2;
uniform int u_shadow_edge_aware_upsampling_1;
uniform int u_shadow_edge_aware_upsampling_2;
uniform float u_shadow_soft_radius_1;
uniform float u_shadow_soft_radius_2;
uniform float u_depth_edge_threshold_m_1;
uniform float u_depth_edge_threshold_m_2;

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

ivec2 shadow_size(int light_index) {
    return light_index == 0
        ? textureSize(u_shadow_visibility_1, 0)
        : textureSize(u_shadow_visibility_2, 0);
}

float shadow_fetch(ivec2 coord, int light_index) {
    return light_index == 0
        ? texelFetch(u_shadow_visibility_1, coord, 0).r
        : texelFetch(u_shadow_visibility_2, coord, 0).r;
}

float shadow_sample(vec2 uv, int light_index) {
    return light_index == 0
        ? texture(u_shadow_visibility_1, uv).r
        : texture(u_shadow_visibility_2, uv).r;
}

int edge_aware(int light_index) {
    return light_index == 0 ? u_shadow_edge_aware_upsampling_1 : u_shadow_edge_aware_upsampling_2;
}

int soft_enabled(int light_index) {
    return light_index == 0 ? u_shadow_softening_enabled_1 : u_shadow_softening_enabled_2;
}

int soft_samples(int light_index) {
    return light_index == 0 ? u_shadow_soft_samples_1 : u_shadow_soft_samples_2;
}

float soft_radius(int light_index) {
    return light_index == 0 ? u_shadow_soft_radius_1 : u_shadow_soft_radius_2;
}

float edge_threshold(int light_index) {
    return light_index == 0 ? u_depth_edge_threshold_m_1 : u_depth_edge_threshold_m_2;
}

float depth_edge_weight(ivec2 shadow_coord, float receiver_z, bool receiver_valid, int light_index) {
    if (edge_aware(light_index) == 0 || !receiver_valid) {
        return 1.0;
    }
    ivec2 size = shadow_size(light_index);
    // Shadow FBO coordinates are bottom-left; uploaded camera depth is top-left.
    vec2 tap_uv = vec2(
        (float(shadow_coord.x) + 0.5) / float(size.x),
        1.0 - (float(shadow_coord.y) + 0.5) / float(size.y)
    );
    float tap_z = texture(u_depth, tap_uv).r;
    float tap_valid = texture(u_depth_valid, tap_uv).r;
    if (tap_valid <= 0.0 || isnan(tap_z) || isinf(tap_z) || tap_z <= 0.0) {
        return 0.0;
    }
    float difference_m = abs(receiver_z - tap_z);
    return max(0.0, 1.0 - difference_m / edge_threshold(light_index));
}

float bilinear_visibility(vec2 shadow_position, float receiver_z, bool receiver_valid, int light_index) {
    ivec2 size = shadow_size(light_index);
    if (edge_aware(light_index) == 0) {
        return shadow_sample((shadow_position + vec2(0.5)) / vec2(size), light_index);
    }

    ivec2 base = ivec2(floor(shadow_position));
    vec2 fraction = fract(shadow_position);
    float visibility_sum = 0.0;
    float weight_sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        ivec2 raw_coord = base + BILINEAR_OFFSETS[i];
        ivec2 coord = clamp(raw_coord, ivec2(0), size - ivec2(1));
        vec2 side = vec2(BILINEAR_OFFSETS[i]);
        vec2 axis_weight = mix(vec2(1.0) - fraction, fraction, side);
        float spatial_weight = axis_weight.x * axis_weight.y;
        float edge_weight = depth_edge_weight(coord, receiver_z, receiver_valid, light_index);
        float weight = spatial_weight * edge_weight;
        visibility_sum += shadow_fetch(coord, light_index) * weight;
        weight_sum += weight;
    }

    // If no tap can be associated with this surface, conservatively leave it lit.
    return weight_sum <= 1e-6 ? 1.0 : visibility_sum / weight_sum;
}

float filtered_visibility(float receiver_z, bool receiver_valid, int light_index) {
    ivec2 size = shadow_size(light_index);
    vec2 shadow_uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec2 shadow_position = shadow_uv * vec2(size) - vec2(0.5);
    float center = bilinear_visibility(shadow_position, receiver_z, receiver_valid, light_index);
    if (soft_enabled(light_index) == 0) {
        return center;
    }

    float visibility_sum = center;
    float weight_sum = 1.0;
    for (int i = 0; i < 8; ++i) {
        if (i >= soft_samples(light_index)) {
            break;
        }
        vec2 offset_position = shadow_position + SOFT_OFFSETS[i] * soft_radius(light_index);
        visibility_sum += bilinear_visibility(offset_position, receiver_z, receiver_valid, light_index);
        weight_sum += 1.0;
    }
    return visibility_sum / weight_sum;
}

void main() {
    // Lighting layers are rendered into bottom-left-origin FBOs.
    vec2 layer_uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec3 ambient = texture(u_ambient, layer_uv).rgb;
    vec3 direct_1 = texture(u_direct_1, layer_uv).rgb;
    vec3 direct_2 = texture(u_direct_2, layer_uv).rgb;
    float receiver_z = texture(u_depth, v_uv).r;
    float receiver_valid_value = texture(u_depth_valid, v_uv).r;
    bool receiver_valid = receiver_valid_value > 0.0
        && !isnan(receiver_z) && !isinf(receiver_z) && receiver_z > 0.0;
    float visibility_1 = clamp(filtered_visibility(receiver_z, receiver_valid, 0), 0.0, 1.0);
    float visibility_2 = clamp(filtered_visibility(receiver_z, receiver_valid, 1), 0.0, 1.0);
    vec3 result_linear = ambient + visibility_1 * direct_1 + visibility_2 * direct_2;
    if (any(isnan(result_linear)) || any(isinf(result_linear))) {
        result_linear = ambient;
    }
    if (u_volumetric_enabled != 0) {
        vec3 volume = texture(u_volumetric, layer_uv).rgb;
        if (!any(isnan(volume)) && !any(isinf(volume))) {
            // Add within available linear-light headroom to avoid washing
            // highlights to white; when disabled the P11 expression is exact.
            vec3 headroom = max(vec3(1.0) - clamp(result_linear, 0.0, 1.0), vec3(0.0));
            result_linear += min(max(volume, vec3(0.0)), headroom * 0.8);
        }
    }
    frag_color = vec4(linear_to_srgb(result_linear), 1.0);
}

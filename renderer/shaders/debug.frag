#version 330

uniform sampler2D u_rgb;
uniform sampler2D u_depth;
uniform sampler2D u_normals;
uniform sampler2D u_depth_valid;
uniform sampler2D u_normal_valid;
uniform sampler2D u_shadow_visibility;
uniform sampler2D u_shadow_visibility_2;
uniform sampler2D u_volumetric;
uniform int u_debug_mode;
uniform float u_depth_min_m;
uniform float u_depth_max_m;
uniform float u_depth_edge_threshold_m;
uniform float u_volumetric_debug_scale;

in vec2 v_uv;
out vec4 frag_color;

vec3 linear_to_srgb(vec3 linear_rgb) {
    vec3 clamped = clamp(linear_rgb, 0.0, 1.0);
    vec3 low = clamped * 12.92;
    vec3 high = 1.055 * pow(clamped, vec3(1.0 / 2.4)) - 0.055;
    return mix(low, high, step(vec3(0.0031308), clamped));
}

void main() {
    vec3 color;
    if (u_debug_mode == 1) {
        color = texture(u_rgb, v_uv).rgb;
    } else if (u_debug_mode == 2) {
        float depth_m = texture(u_depth, v_uv).r;
        float valid = texture(u_depth_valid, v_uv).r;
        float depth_gray = valid > 0.0 && depth_m > 0.0 && !isnan(depth_m) && !isinf(depth_m)
            ? clamp((u_depth_max_m - depth_m) / (u_depth_max_m - u_depth_min_m), 0.0, 1.0)
            : 0.0;
        color = vec3(depth_gray);
    } else if (u_debug_mode == 7) {
        // Shadow mask convention: white = illuminated, black = shadowed.
        // This texture is a framebuffer output, so its Y origin is bottom-left.
        float visibility = texture(u_shadow_visibility, vec2(v_uv.x, 1.0 - v_uv.y)).r;
        color = vec3(clamp(visibility, 0.0, 1.0));
    } else if (u_debug_mode == 10) {
        float visibility = texture(u_shadow_visibility_2, vec2(v_uv.x, 1.0 - v_uv.y)).r;
        color = vec3(clamp(visibility, 0.0, 1.0));
    } else if (u_debug_mode == 11) {
        vec3 contribution = texture(u_volumetric, vec2(v_uv.x, 1.0 - v_uv.y)).rgb;
        color = linear_to_srgb(max(contribution * u_volumetric_debug_scale, vec3(0.0)));
    } else if (u_debug_mode == 9) {
        float center_z = texture(u_depth, v_uv).r;
        bool center_valid = texture(u_depth_valid, v_uv).r > 0.0
            && !isnan(center_z) && !isinf(center_z) && center_z > 0.0;
        float edge_delta = 0.0;
        if (center_valid) {
            vec2 texel = 1.0 / vec2(textureSize(u_depth, 0));
            const vec2 directions[4] = vec2[4](
                vec2(1.0, 0.0), vec2(-1.0, 0.0),
                vec2(0.0, 1.0), vec2(0.0, -1.0)
            );
            for (int i = 0; i < 4; ++i) {
                vec2 sample_uv = clamp(v_uv + directions[i] * texel, vec2(0.0), vec2(1.0));
                float sample_z = texture(u_depth, sample_uv).r;
                bool sample_valid = texture(u_depth_valid, sample_uv).r > 0.0
                    && !isnan(sample_z) && !isinf(sample_z) && sample_z > 0.0;
                if (sample_valid) {
                    edge_delta = max(edge_delta, abs(center_z - sample_z));
                }
            }
        }
        color = vec3(clamp(edge_delta / u_depth_edge_threshold_m, 0.0, 1.0));
    } else {
        vec3 normal = texture(u_normals, v_uv).xyz;
        float magnitude = length(normal);
        bool valid = texture(u_normal_valid, v_uv).r > 0.0;
        color = valid && magnitude > 0.0001 && !any(isnan(normal)) && !any(isinf(normal))
            ? normalize(normal) * 0.5 + 0.5
            : vec3(0.0);
    }
    frag_color = vec4(color, 1.0);
}

#version 330

uniform sampler2D u_ambient;
uniform sampler2D u_direct;
uniform sampler2D u_shadow_visibility;

in vec2 v_uv;
out vec4 frag_color;

vec3 linear_to_srgb(vec3 linear_rgb) {
    vec3 clamped = clamp(linear_rgb, 0.0, 1.0);
    vec3 low = clamped * 12.92;
    vec3 high = 1.055 * pow(clamped, vec3(1.0 / 2.4)) - 0.055;
    return mix(low, high, step(vec3(0.0031308), clamped));
}

void main() {
    vec3 ambient = texture(u_ambient, v_uv).rgb;
    vec3 direct = texture(u_direct, v_uv).rgb;
    float visibility = clamp(texture(u_shadow_visibility, v_uv).r, 0.0, 1.0);
    vec3 result_linear = ambient + visibility * direct;
    if (any(isnan(result_linear)) || any(isinf(result_linear))) {
        result_linear = ambient;
    }
    frag_color = vec4(linear_to_srgb(result_linear), 1.0);
}

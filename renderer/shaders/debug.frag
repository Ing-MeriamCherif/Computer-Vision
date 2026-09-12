#version 330

uniform sampler2D u_rgb;
uniform sampler2D u_depth;
uniform sampler2D u_normals;
uniform int u_debug_mode;
uniform float u_depth_min_m;
uniform float u_depth_max_m;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 color;
    if (u_debug_mode == 1) {
        color = texture(u_rgb, v_uv).rgb;
    } else if (u_debug_mode == 2) {
        float depth_m = texture(u_depth, v_uv).r;
        float depth_gray = depth_m > 0.0
            ? clamp((u_depth_max_m - depth_m) / (u_depth_max_m - u_depth_min_m), 0.0, 1.0)
            : 0.0;
        color = vec3(depth_gray);
    } else {
        vec3 normal = texture(u_normals, v_uv).xyz;
        float magnitude = length(normal);
        color = magnitude > 0.0001 ? normalize(normal) * 0.5 + 0.5 : vec3(0.0);
    }
    frag_color = vec4(color, 1.0);
}

#version 330

uniform sampler2D u_depth;
uniform sampler2D u_depth_valid;
uniform float u_fx;
uniform float u_fy;
uniform float u_cx;
uniform float u_cy;
uniform float u_image_width;
uniform float u_image_height;
uniform vec3 u_light_position_camera_m;
uniform int u_light_active;
uniform int u_shadow_steps;
uniform float u_shadow_bias_m;
uniform float u_shadow_thickness_m;
uniform float u_ray_start_offset;

in vec2 v_uv;
out vec4 frag_color;

bool finite_vec3(vec3 value) {
    return !any(isnan(value)) && !any(isinf(value));
}

float unblocked() {
    // The shadow mask convention is white = visible/unblocked.
    return 1.0;
}

void main() {
    if (u_light_active == 0 || !finite_vec3(u_light_position_camera_m)) {
        frag_color = vec4(vec3(unblocked()), 1.0);
        return;
    }

    vec2 image_size = vec2(u_image_width, u_image_height);
    vec2 pixel = v_uv * image_size - vec2(0.5);
    float receiver_z = texture(u_depth, v_uv).r;
    float receiver_valid = texture(u_depth_valid, v_uv).r;
    if (receiver_valid <= 0.0 || isnan(receiver_z) || isinf(receiver_z) || receiver_z <= 0.0) {
        frag_color = vec4(vec3(unblocked()), 1.0);
        return;
    }

    vec3 point = vec3(
        (pixel.x - u_cx) * receiver_z / u_fx,
        (pixel.y - u_cy) * receiver_z / u_fy,
        receiver_z
    );
    vec3 ray = u_light_position_camera_m - point;
    float ray_length = length(ray);
    if (!finite_vec3(point) || isnan(ray_length) || isinf(ray_length) || ray_length <= 1e-6) {
        frag_color = vec4(vec3(unblocked()), 1.0);
        return;
    }

    // Offset in meters along the ray, then fixed samples strictly inside (P, light).
    float start_t = clamp(u_ray_start_offset / ray_length, 0.0, 0.99);
    for (int step_index = 0; step_index < 64; ++step_index) {
        if (step_index >= u_shadow_steps) {
            break;
        }
        float t = start_t + (1.0 - start_t) * float(step_index + 1)
            / float(u_shadow_steps + 1);
        vec3 sample_point = point + t * ray;
        if (!finite_vec3(sample_point) || sample_point.z <= 1e-6) {
            frag_color = vec4(vec3(unblocked()), 1.0);
            return;
        }

        vec2 sample_pixel = vec2(
            u_fx * sample_point.x / sample_point.z + u_cx,
            u_fy * sample_point.y / sample_point.z + u_cy
        );
        if (any(lessThan(sample_pixel, vec2(0.0)))
            || any(greaterThan(sample_pixel, image_size - vec2(1.0)))) {
            // Screen-space cannot see off-screen geometry; unknown is unblocked.
            frag_color = vec4(vec3(unblocked()), 1.0);
            return;
        }

        vec2 sample_uv = (sample_pixel + vec2(0.5)) / image_size;
        float scene_z = texture(u_depth, sample_uv).r;
        float sample_valid = texture(u_depth_valid, sample_uv).r;
        if (sample_valid <= 0.0 || isnan(scene_z) || isinf(scene_z) || scene_z <= 0.0) {
            // Unknown depth never creates a black shadow.
            frag_color = vec4(vec3(unblocked()), 1.0);
            return;
        }

        // Count a hit only when bias < ray_z - scene_z <= thickness. A sample
        // within the bias is treated as the receiver itself; a larger gap than
        // thickness is rejected as an ambiguous depth discontinuity/step skip.
        float behind_surface_m = sample_point.z - scene_z;
        if (behind_surface_m > u_shadow_bias_m
            && behind_surface_m <= u_shadow_thickness_m) {
            frag_color = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
    }

    frag_color = vec4(vec3(unblocked()), 1.0);
}

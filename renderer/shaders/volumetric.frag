#version 330

uniform sampler2D u_depth;
uniform sampler2D u_depth_valid;
uniform sampler2D u_shadow_visibility_1;
uniform sampler2D u_shadow_visibility_2;
uniform float u_fx;
uniform float u_fy;
uniform float u_cx;
uniform float u_cy;
uniform float u_image_width;
uniform float u_image_height;
uniform int u_light_count;
uniform vec3 u_light_positions_camera_m[2];
uniform vec3 u_light_colors_rgb[2];
uniform float u_light_intensities[2];
uniform int u_light_active[2];
uniform float u_attenuation_k;
uniform int u_volumetric_samples;
uniform float u_volumetric_density;
uniform float u_volumetric_intensity;
uniform float u_volumetric_decay;

in vec2 v_uv;
out vec4 frag_color;

const int MAX_LIGHTS = 2;
const int MAX_SAMPLES = 24;

bool finite_vec3(vec3 value) {
    return !any(isnan(value)) && !any(isinf(value));
}

float light_visibility(vec2 image_uv, int light_index) {
    // Shadow maps describe visibility at the visible surface. Reuse that
    // value as a low-cost approximation for every sample on this view ray.
    vec2 shadow_uv = vec2(image_uv.x, 1.0 - image_uv.y);
    return light_index == 0
        ? texture(u_shadow_visibility_1, shadow_uv).r
        : texture(u_shadow_visibility_2, shadow_uv).r;
}

void main() {
    float surface_z = texture(u_depth, v_uv).r;
    float depth_valid = texture(u_depth_valid, v_uv).r;
    if (depth_valid <= 0.0 || isnan(surface_z) || isinf(surface_z)
        || surface_z <= 0.0 || surface_z > 1000.0) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec2 pixel = v_uv * vec2(u_image_width, u_image_height) - vec2(0.5);
    vec3 ray = vec3(
        (pixel.x - u_cx) / u_fx,
        (pixel.y - u_cy) / u_fy,
        1.0
    );
    float ray_length = length(ray);
    if (isnan(ray_length) || isinf(ray_length) || ray_length <= 1e-6) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    ray /= ray_length;

    // Metric depth is camera Z, so divide by the ray's Z component to get
    // distance along the normalized ray to the first visible surface.
    float ray_end_m = surface_z / max(ray.z, 1e-6);
    float step_m = ray_end_m / float(u_volumetric_samples);
    float decay_weight = 1.0;
    vec3 scattering = vec3(0.0);

    for (int sample_index = 0; sample_index < MAX_SAMPLES; ++sample_index) {
        if (sample_index >= u_volumetric_samples) {
            break;
        }
        float distance_along_ray = (float(sample_index) + 0.5) * step_m;
        vec3 sample_position = ray * distance_along_ray;
        for (int light_index = 0; light_index < MAX_LIGHTS; ++light_index) {
            if (light_index >= u_light_count || u_light_active[light_index] == 0) {
                continue;
            }
            vec3 light_position = u_light_positions_camera_m[light_index];
            vec3 light_color = u_light_colors_rgb[light_index];
            float light_intensity = u_light_intensities[light_index];
            if (!finite_vec3(light_position) || !finite_vec3(light_color)
                || isnan(light_intensity) || isinf(light_intensity) || light_intensity <= 0.0) {
                continue;
            }

            vec3 to_light = light_position - sample_position;
            float distance_sq = dot(to_light, to_light);
            if (isnan(distance_sq) || isinf(distance_sq)) {
                continue;
            }
            float attenuation = 1.0 / (1.0 + u_attenuation_k * max(distance_sq, 0.0));
            float visibility = clamp(light_visibility(v_uv, light_index), 0.0, 1.0);
            float integration_weight = u_volumetric_density * step_m
                * u_volumetric_intensity * decay_weight;
            scattering += clamp(light_color, 0.0, 1.0)
                * light_intensity * attenuation * visibility * integration_weight;
        }
        decay_weight *= u_volumetric_decay;
    }

    if (!finite_vec3(scattering)) {
        scattering = vec3(0.0);
    }
    // Keep values representable in the persistent RGB16F target. Composition
    // applies a stricter per-pixel headroom limit before adding this term.
    frag_color = vec4(clamp(scattering, vec3(0.0), vec3(16.0)), 1.0);
}

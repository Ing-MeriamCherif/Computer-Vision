#version 330

uniform sampler2D u_rgb;
uniform sampler2D u_depth;
uniform sampler2D u_normals;
uniform sampler2D u_depth_valid;
uniform sampler2D u_normal_valid;

uniform int u_lighting_mode; // 4 = diffuse, 5 = specular, 6 = Level 02 final, 8 = linear lighting layers
uniform float u_fx;
uniform float u_fy;
uniform float u_cx;
uniform float u_cy;
uniform float u_image_width;
uniform float u_image_height;

uniform vec3 u_light_position_camera_m;
uniform vec3 u_light_color_rgb;
uniform float u_light_intensity;
uniform int u_light_active;

uniform float u_ambient_strength;
uniform float u_specular_strength;
uniform float u_shininess;
uniform float u_attenuation_k;

in vec2 v_uv;
layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec4 direct_color;

bool finite_vec3(vec3 value) {
    return !any(isnan(value)) && !any(isinf(value));
}

vec3 srgb_to_linear(vec3 srgb) {
    // Input RGB8 camera/synthetic bytes are assumed to be sRGB encoded.
    vec3 low = srgb / 12.92;
    vec3 high = pow((srgb + 0.055) / 1.055, vec3(2.4));
    return mix(low, high, step(vec3(0.04045), srgb));
}

vec3 linear_to_srgb(vec3 linear_rgb) {
    vec3 clamped = clamp(linear_rgb, 0.0, 1.0);
    vec3 low = clamped * 12.92;
    vec3 high = 1.055 * pow(clamped, vec3(1.0 / 2.4)) - 0.055;
    return mix(low, high, step(vec3(0.0031308), clamped));
}

void output_source(vec3 source_srgb) {
    if (u_lighting_mode == 8) {
        // Invalid geometry remains visible and contributes no direct light.
        frag_color = vec4(srgb_to_linear(clamp(source_srgb, 0.0, 1.0)), 1.0);
        direct_color = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        frag_color = vec4(source_srgb, 1.0);
        direct_color = vec4(0.0, 0.0, 0.0, 1.0);
    }
}

void main() {
    vec3 source_srgb = texture(u_rgb, v_uv).rgb;
    if (!finite_vec3(source_srgb)) {
        source_srgb = vec3(0.0);
    }

    float z_m = texture(u_depth, v_uv).r;
    // R8 normalized validity textures encode True as byte 1 (about 0.0039).
    bool depth_valid = texture(u_depth_valid, v_uv).r > 0.0;
    if (!depth_valid || isnan(z_m) || isinf(z_m) || z_m <= 1e-6) {
        // Unknown geometry keeps the original image in all lighting views.
        output_source(source_srgb);
        return;
    }

    vec3 normal_camera = texture(u_normals, v_uv).xyz;
    float normal_length = length(normal_camera);
    bool normal_valid = texture(u_normal_valid, v_uv).r > 0.0;
    if (!normal_valid || !finite_vec3(normal_camera) || isnan(normal_length)
        || isinf(normal_length) || normal_length <= 1e-6) {
        output_source(source_srgb);
        return;
    }

    vec2 pixel = v_uv * vec2(u_image_width, u_image_height) - vec2(0.5);
    vec3 point_camera_m = vec3(
        (pixel.x - u_cx) * z_m / u_fx,
        (pixel.y - u_cy) * z_m / u_fy,
        z_m
    );
    if (!finite_vec3(point_camera_m) || !finite_vec3(u_light_position_camera_m)) {
        output_source(source_srgb);
        return;
    }

    vec3 albedo_linear = srgb_to_linear(clamp(source_srgb, 0.0, 1.0));
    vec3 normal = normal_camera / normal_length;
    vec3 ambient = albedo_linear * u_ambient_strength;
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);

    if (u_light_active != 0 && u_light_intensity > 0.0) {
        vec3 light_vector = u_light_position_camera_m - point_camera_m;
        float distance_m = length(light_vector);
        if (!isnan(distance_m) && !isinf(distance_m) && distance_m > 1e-6) {
            vec3 light_direction = light_vector / distance_m;
            float attenuation = 1.0 / (1.0 + u_attenuation_k * distance_m * distance_m);
            float diffuse_factor = max(dot(normal, light_direction), 0.0);
            diffuse = albedo_linear * u_light_color_rgb * u_light_intensity
                * attenuation * diffuse_factor;

            if (diffuse_factor > 0.0) {
                vec3 view_vector = -point_camera_m;
                float view_length = length(view_vector);
                if (view_length > 1e-6 && !isnan(view_length) && !isinf(view_length)) {
                    vec3 view_direction = view_vector / view_length;
                    vec3 half_vector = light_direction + view_direction;
                    float half_length = length(half_vector);
                    if (half_length > 1e-6 && !isnan(half_length) && !isinf(half_length)) {
                        vec3 half_direction = half_vector / half_length;
                        float specular_factor = pow(
                            max(dot(normal, half_direction), 0.0),
                            u_shininess
                        );
                        specular = u_light_color_rgb * u_light_intensity * attenuation
                            * u_specular_strength * specular_factor;
                    }
                }
            }
        }
    }

    vec3 result_linear;
    if (u_lighting_mode == 8) {
        // MRT outputs are kept linear so ambient can remain unshadowed during composition.
        frag_color = vec4(ambient, 1.0);
        direct_color = vec4(diffuse + specular, 1.0);
        return;
    } else if (u_lighting_mode == 4) {
        result_linear = ambient + diffuse;
    } else if (u_lighting_mode == 5) {
        result_linear = specular;
    } else {
        result_linear = ambient + diffuse + specular;
    }

    if (!finite_vec3(result_linear)) {
        output_source(source_srgb);
        return;
    }
    frag_color = vec4(linear_to_srgb(result_linear), 1.0);
    direct_color = vec4(0.0, 0.0, 0.0, 1.0);
}

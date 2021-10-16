#version 450

layout (location = 0) in vec3 v_pos;
layout (location = 1) in vec3 v_norm;
layout (location = 2) in vec4 v_tan;
layout (location = 3) in vec2 v_uv;

layout (location = 0) out vec4 o_base_color_emissive;
layout (location = 1) out vec4 o_pos_roughness;
layout (location = 2) out vec4 o_norm_metallic;
layout (location = 3) out uint o_id;

layout (binding = 1) uniform DrawableUniforms {
    mat4 u_model;
    mat4 u_model_it;
    uint u_id;
};

layout (binding = 2) uniform MaterialUniforms {
    vec4 base_color;
    vec3 emissive;
    float metallic;
    float roughness;
    int alpha_mode;
    float alpha_cutoff;
    float normal_tex_scale;
} mat;

layout (binding = 3) uniform sampler2D base_color_tex;

layout (binding = 4) uniform sampler2D emissive_tex;

layout (binding = 5) uniform sampler2D metallic_roughness_tex;

layout (binding = 6) uniform sampler2D normal_tex;

void main() {
    o_id = u_id;
    o_pos_roughness.xyz = v_pos;

    const vec4 base_color = texture(base_color_tex, v_uv) * mat.base_color;
    if (mat.alpha_mode == 1 && base_color.a < mat.alpha_cutoff) {
        discard;
    }

    const vec3 bitan = cross(v_norm, v_tan.xyz) * v_tan.w;
    const vec3 norm_tex_value = (texture(normal_tex, v_uv).xyz * 2.0 - 1.0);
    const vec3 scaled_norm = normalize(norm_tex_value * vec3(mat.normal_tex_scale, mat.normal_tex_scale, 1.0));
    o_norm_metallic.xyz = normalize(scaled_norm.x * v_tan.xyz + scaled_norm.y * bitan + scaled_norm.z * v_norm);

    const vec4 mr_tex = texture(metallic_roughness_tex, v_uv);
    const float roughness = mat.roughness * mr_tex.y;
    const float metallic = mat.metallic * mr_tex.z;
    o_pos_roughness.w = roughness;
    o_norm_metallic.w = metallic;

    if (dot(mat.emissive, mat.emissive) > 0.0) {
        o_base_color_emissive.xyz = mat.emissive * texture(emissive_tex, v_uv).rgb;
        o_base_color_emissive.w = 1.0;
    } else {
        o_base_color_emissive.xyz = base_color.rgb;
        o_base_color_emissive.w = 0.0;
    }
}
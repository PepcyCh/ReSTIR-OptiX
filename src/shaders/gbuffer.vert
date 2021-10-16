#version 450

layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_norm;
layout (location = 2) in vec4 a_tan;
layout (location = 3) in vec2 a_uv;

layout (location = 0) out vec3 v_pos;
layout (location = 1) out vec3 v_norm;
layout (location = 2) out vec4 v_tan;
layout (location = 3) out vec2 v_uv;

layout (binding = 0) uniform CameraUniforms {
    mat4 u_proj_view;
};

layout (binding = 1) uniform DrawableUniforms {
    mat4 u_model;
    mat4 u_model_it;
    uint u_id;
};

void main() {
    const vec4 world_pos = u_model * vec4(a_pos, 1.0);
    v_pos = world_pos.xyz;
    gl_Position = u_proj_view * world_pos;

    v_norm = normalize(mat3(u_model_it) * a_norm);
    v_tan.xyz = normalize(mat3(u_model) * a_tan.xyz);
    v_tan.w = a_tan.w;

    v_uv = a_uv;
}
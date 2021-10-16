#version 450

layout (location = 0) out vec2 v_texcoords;

void main() {
    v_texcoords = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(
        v_texcoords.x * 2.0 - 1.0,
        1.0 - v_texcoords.y * 2.0,
        1.0,
        1.0
    );
}
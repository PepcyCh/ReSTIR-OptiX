#version 450

layout (location = 0) in vec2 v_texcoords;

layout (location = 0) out vec4 o_frag_color;

layout (binding = 0) uniform sampler2D u_img;

void main() {
    vec4 color = texture(u_img, v_texcoords);
    o_frag_color = color;
}
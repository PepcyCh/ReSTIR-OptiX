#include "gl_utils/framebuffer.hpp"

#include "glad/gl.h"

GlFramebuffer::~GlFramebuffer() {
    Delete();
}

void GlFramebuffer::CreateAsGBuffer(int width, int height) {
    glCreateFramebuffers(1, &id);

    num_color_attachments = 4;
    color_attachments[0].Create(width, height, GL_RGBA32F); // albedo | is emmisive
    color_attachments[1].Create(width, height, GL_RGBA32F); // pos | roughness
    color_attachments[2].Create(width, height, GL_RGBA32F); // norm | metallic
    color_attachments[3].Create(width, height, GL_R32UI); // id
    glNamedFramebufferTexture(id, GL_COLOR_ATTACHMENT0, color_attachments[0].id, 0);
    glNamedFramebufferTexture(id, GL_COLOR_ATTACHMENT1, color_attachments[1].id, 0);
    glNamedFramebufferTexture(id, GL_COLOR_ATTACHMENT2, color_attachments[2].id, 0);
    glNamedFramebufferTexture(id, GL_COLOR_ATTACHMENT3, color_attachments[3].id, 0);
    const GLenum color_attachments[] = {
        GL_COLOR_ATTACHMENT0,
        GL_COLOR_ATTACHMENT1,
        GL_COLOR_ATTACHMENT2,
        GL_COLOR_ATTACHMENT3
    };
    glNamedFramebufferDrawBuffers(id, 4, color_attachments);

    has_depth_stencil_attachment = true;
    depth_stencil_attachement.Create(width, height, GL_DEPTH_COMPONENT32F);
    glNamedFramebufferTexture(id, GL_DEPTH_ATTACHMENT, depth_stencil_attachement.id, 0);
}

void GlFramebuffer::Delete() {
    if (id) {
        glDeleteFramebuffers(1, &id);
        id = 0;
        for (uint8_t i = 0; i < num_color_attachments; i++) {
            color_attachments->Delete();
        }
        num_color_attachments = 0;
        if (has_depth_stencil_attachment) {
            depth_stencil_attachement.Delete();
        }
        has_depth_stencil_attachment = false;
    }
}
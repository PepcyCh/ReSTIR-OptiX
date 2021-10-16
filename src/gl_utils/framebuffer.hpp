#pragma once

#include "gl_utils/texture.hpp"

struct GlFramebuffer {
    ~GlFramebuffer();

    void CreateAsGBuffer(int width, int height);

    void Delete();

    uint32_t id;
    uint8_t num_color_attachments;
    bool has_depth_stencil_attachment;
    GlTexture2D color_attachments[8];
    GlTexture2D depth_stencil_attachement;
};
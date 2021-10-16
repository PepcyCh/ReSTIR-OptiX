#include "gl_utils/texture.hpp"

#include <cassert>

#include "glad/gl.h"

namespace {

void GetFormatChannelFormat(GLenum format, GLenum &channel_format, GLenum &channel_type) {
    switch (format) {
    case GL_RGBA8:
        channel_format = GL_RGBA;
        channel_type = GL_UNIFORM_TYPE;
        break;
    case GL_RGBA32F:
        channel_format = GL_RGBA;
        channel_type = GL_FLOAT;
        break;
    case GL_RGB8:
        channel_format = GL_RGB;
        channel_type = GL_UNIFORM_TYPE;
        break;
    case GL_RGB32F:
        channel_format = GL_RGB;
        channel_type = GL_FLOAT;
        break;
    case GL_DEPTH_COMPONENT32F:
        channel_format = GL_DEPTH_COMPONENT;
        channel_type = GL_FLOAT;
        break;
    case GL_DEPTH24_STENCIL8:
        channel_format = GL_DEPTH_STENCIL;
        channel_type = GL_UNSIGNED_INT_24_8;
        break;
    case GL_DEPTH32F_STENCIL8:
        channel_format = GL_DEPTH_STENCIL;
        channel_type = GL_FLOAT_32_UNSIGNED_INT_24_8_REV;
        break;
    case GL_R32UI:
        channel_format = GL_RED_INTEGER;
        channel_type = GL_UNSIGNED_INT;
        break;
    default:
        // just list what I used ...
        assert("Unknown texture format" && false);
    }
}

}

GlTexture2D::~GlTexture2D() {
    Delete();
}

void GlTexture2D::Create(int width, int height, uint32_t format) {
    this->width = width;
    this->height = height;
    this->format = format;
    glCreateTextures(GL_TEXTURE_2D, 1, &id);
    glTextureStorage2D(id, 1, format, width, height);
    GetFormatChannelFormat(format, channel_format, channel_type);
    if (channel_type != GL_UNSIGNED_INT) {
        glTextureParameteri(id, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    } else {
        glTextureParameteri(id, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
}

void GlTexture2D::Delete() {
    if (id) {
        glDeleteTextures(1, &id);
        id = 0;
    }
}
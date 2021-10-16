#include "cuda_utils/interop_buffer.hpp"

#include <cassert>

#include "glad/gl.h" // glad header must be put ahead of other gl headers
#include "cuda_gl_interop.h"

#include "misc/check_macros.hpp"

InteropBuffer::~InteropBuffer() {
    cudaGraphicsUnregisterResource(cuda_res_);
    glDeleteBuffers(1, &gl_buffer_);
}

void InteropBuffer::SwapWith(InteropBuffer &other) {
    std::swap(gl_buffer_, other.gl_buffer_);
    std::swap(cuda_res_, other.cuda_res_);
    std::swap(size_, other.size_);
}

void InteropBuffer::Init(size_t size, uint32_t flag) {
    size_ = size;

    glCreateBuffers(1, &gl_buffer_);
    glNamedBufferStorage(gl_buffer_, size, nullptr, 0);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_res_, gl_buffer_, flag));
}

void InteropBuffer::Resize(size_t size, uint32_t flag) {
    if (cuda_res_) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_res_));
        glDeleteBuffers(1, &gl_buffer_);
    }

    Init(size, flag);
}

void *InteropBuffer::Map() {
    assert(gl_buffer_ && cuda_res_);

    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_res_));
    size_t mapped_size;
    void *ptr;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&ptr, &mapped_size, cuda_res_));
    return ptr;
}

void InteropBuffer::Unmap() {
    assert(gl_buffer_ && cuda_res_);

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_res_));
}

void InteropBuffer::UnpackTo(const GlTexture2D &tex) {
    assert(gl_buffer_ && cuda_res_);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_buffer_);
    glTextureSubImage2D(tex.id, 0, 0, 0, tex.width, tex.height, tex.channel_format, tex.channel_type, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void InteropBuffer::PackFrom(const GlFramebuffer &fb, uint32_t attachment) {
    assert(gl_buffer_ && cuda_res_);

    glNamedFramebufferReadBuffer(fb.id, attachment);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, gl_buffer_);
    if (attachment >= GL_COLOR_ATTACHMENT0 && attachment <= GL_COLOR_ATTACHMENT7) {
        const GlTexture2D &tex = fb.color_attachments[attachment - GL_COLOR_ATTACHMENT0];
        glReadPixels(0, 0, tex.width, tex.height, tex.channel_format, tex.channel_type, nullptr);
    } else if (attachment == GL_DEPTH_ATTACHMENT || attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
        const GlTexture2D &tex = fb.depth_stencil_attachement;
        glReadPixels(0, 0, tex.width, tex.height, tex.channel_format, tex.channel_type, nullptr);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}
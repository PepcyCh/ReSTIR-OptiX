#pragma once

#include <cstdint>

#include "texture_types.h"

#include "gl_utils/framebuffer.hpp"

class InteropBuffer {
public:
    ~InteropBuffer();

    void SwapWith(InteropBuffer &other);

    void Init(size_t size, uint32_t flag);

    void Resize(size_t size, uint32_t flag);

    void *Map();

    template <typename T>
    T *TypedMap() {
        return static_cast<T *>(Map());
    }

    void Unmap();

    void UnpackTo(const GlTexture2D &tex);

    void PackFrom(const GlFramebuffer &fb, uint32_t attachment);

private:
    uint32_t gl_buffer_ = 0;
    cudaGraphicsResource_t cuda_res_ = nullptr;

    size_t size_ = 0;
};
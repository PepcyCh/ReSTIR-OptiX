#pragma once

#include <cstdint>

struct GlTexture2D {
    ~GlTexture2D();

    void Create(int width, int height, uint32_t format);

    void Delete();

    uint32_t id = 0;
    int width = 0;
    int height = 0;
    uint32_t format = 0;
    uint32_t channel_format = 0;
    uint32_t channel_type = 0;
};
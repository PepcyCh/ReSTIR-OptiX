#pragma once

#include <unordered_map>
#include <string>

struct GlProgram {
    GlProgram(const std::string &vs, const std::string &fs);
    ~GlProgram();

    uint32_t id;

private:
    inline static std::unordered_map<std::string, uint32_t> cached_shaders;
};
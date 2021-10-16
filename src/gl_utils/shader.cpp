#include "gl_utils/shader.hpp"

#include <fstream>

#include "glad/gl.h"
#include "fmt/core.h"

namespace {

GLuint LoadShader(const std::string &path, GLenum stage) {
    std::ifstream fin(path);
    const std::string code((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());

    GLuint id = glCreateShader(stage);
    const char *p_code = code.c_str();
    glShaderSource(id, 1, &p_code, nullptr);
    glCompileShader(id);

    int ret;
    glGetShaderiv(id, GL_COMPILE_STATUS, &ret);
    if (!ret) {
        char log_info[512];
        glGetShaderInfoLog(id, 512, nullptr, log_info);
        std::string sh_str;
        if (stage == GL_VERTEX_SHADER) {
            sh_str = "vertex";
        } else if (stage == GL_FRAGMENT_SHADER) {
            sh_str = "fragment";
        }
        fmt::print(stderr, "CE on {} shader: {}\n", sh_str, log_info);
        glDeleteShader(id);
        id = 0;
    }

    return id;
}

}

GlProgram::GlProgram(const std::string &vs, const std::string &fs) {
    id = glCreateProgram();

    auto vs_it = cached_shaders.find(vs);
    GLuint vs_id;
    if (vs_it == cached_shaders.end()) {
        vs_id = LoadShader(vs, GL_VERTEX_SHADER);
        cached_shaders[vs] = vs_id;
    } else {
        vs_id = vs_it->second;
    }

    auto fs_it = cached_shaders.find(fs);
    GLuint fs_id;
    if (fs_it == cached_shaders.end()) {
        fs_id = LoadShader(fs, GL_FRAGMENT_SHADER);
        cached_shaders[fs] = fs_id;
    } else {
        fs_id = fs_it->second;
    }

    glAttachShader(id, vs_id);
    glAttachShader(id, fs_id);
    glLinkProgram(id);

    int ret;
    glGetProgramiv(id, GL_LINK_STATUS, &ret);
    if (!ret) {
        char log_info[512];
        glGetProgramInfoLog(id, 512, nullptr, log_info);
        fmt::print(stderr, "Link error: {}\n", log_info);
    }
}

GlProgram::~GlProgram() {
    glDeleteProgram(id);
}
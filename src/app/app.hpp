#pragma once

#include <string>

#include "cuda.h"
#include "GLFW/glfw3.h"
#include "optix.h"

#include "cuda_utils/cuda_buffer.hpp"
#include "cuda_utils/interop_buffer.hpp"
#include "cuda_utils/optix_sbt.hpp"
#include "gl_utils/shader.hpp"
#include "gl_utils/uniforms.hpp"
#include "misc/camera.hpp"
#include "misc/frame_counter.hpp"
#include "misc/lights.hpp"
#include "raytracing/launch_params.hpp"
#include "scene/scene.hpp"

struct RestirAppConfig {
    int width;
    int height;
    std::string title;
    std::string scene_path;
};

class RestirApp {
public:
    RestirApp(const RestirAppConfig &config);
    ~RestirApp();

    RestirApp(const RestirApp &rhs) = delete;
    RestirApp &operator=(const RestirApp &rhs) = delete;

    void MainLoop();

private:
    void InitializeBasic(const RestirAppConfig &config);
    void InitializeGl();
    void InitializeOptix();
    void InitializeScene(const RestirAppConfig &config);
    void InitializeAccel();
    void InitializeSbt();

    void Update();
    void Render();
    void AfterRender();

    void ImguiConfigRestir();

    void Resize(int width, int height);
    void OnMouse(double x, double y, uint8_t state);

    friend void GlfwWindowResizeFunc(GLFWwindow *window, int width, int height);
    friend void GlfwCursorPosFunc(GLFWwindow *window, double x, double y);

    // basic
    CUcontext cuda_context_;
    CUstream cuda_stream_;
    OptixDeviceContext optix_context_;

    GLFWwindow *window_;
    int window_width_;
    int window_height_;

    InteropBuffer color_buffer_;
    GlTexture2D color_buffer_tex_;

    FrameCounter frame_counter_;
    double last_mouse_x_;
    double last_mouse_y_;

    // gl
    std::unique_ptr<GlProgram> gbuffer_shader_;
    std::unique_ptr<GlFramebuffer> gbuffer_fb_;
    uint32_t gbuffer_vao_;
    InteropBuffer gbuffer_base_color_emissive_;
    InteropBuffer gbuffer_pos_roughness_;
    InteropBuffer gbuffer_norm_metallic_;
    InteropBuffer gbuffer_id_;
    InteropBuffer gbuffer_prev_id_;

    std::unique_ptr<GlProgram> screen_blit_shader_;
    uint32_t screen_blit_vao_;

    // optix
    std::unique_ptr<OptixProgram> program_sample_;
    std::unique_ptr<OptixProgram> program_spatial_;
    std::unique_ptr<OptixProgram> program_lighting_;

    std::unique_ptr<OptixSbt> sbt_sample_;
    std::unique_ptr<OptixSbt> sbt_spatial_;
    std::unique_ptr<OptixSbt> sbt_lighting_;

    LaunchParams optix_launch_params_;
    CudaBuffer optix_launch_params_buffer_;

    // scene
    std::unique_ptr<Scene> scene_;
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<Lights> scene_lights_;

    CudaBuffer scene_position_buffer_;
    CudaBuffer scene_normal_buffer_;
    CudaBuffer scene_index_buffer_;
    CudaBuffer scene_drawables_model_buffer_;
    OptixTraversableHandle scene_accel_handle_;
    CudaBuffer scene_accel_buffer_;

    CudaBuffer scene_lights_buffer_;

    uint32_t scene_gl_vb_pos_;
    uint32_t scene_gl_vb_norm_;
    uint32_t scene_gl_vb_tan_;
    uint32_t scene_gl_vb_uv_;
    uint32_t scene_gl_ib_;

    CameraUniforms camera_uniforms_;
    bool camera_uniforms_dirty_;
    std::vector<DrawableUniforms> drawable_uniforms_;
    std::vector<MaterialUniforms> material_uniforms_;
    uint32_t camera_uniforms_buffer_;
    std::vector<uint32_t> drawable_uniforms_buffer_;
    std::vector<uint32_t> material_uniforms_buffer_;

    std::vector<uint32_t> scene_textures_;
    uint32_t default_tex_white_;
    uint32_t default_tex_norm_;

    // restir
    RestirConfig restir_config_;
    CudaBuffer reservoirs_buffer_[2];
    size_t reservoirs_buffer_curr_index_;
    bool reservoirs_buffer_prev_valid_;
};
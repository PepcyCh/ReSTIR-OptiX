#include "glad/gl.h" // glad header must be put ahead of glfw header

#include "app/app.hpp"

#include <fstream>
#include <iostream>
#include <numbers>

#include "cuda_runtime.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "optix_stubs.h"
#include "optix_function_table_definition.h"

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include "restir_optix.hpp"
#include "misc/check_macros.hpp"

namespace {

void OptixLogFunc(uint32_t level, const char *tag, const char *msg, void *) {
    fmt::print(stderr, "OptiX Log: [{}][{}]: {}\n", level, tag, msg);
}

void GlfwErrorLogFunc(int error, const char *desc) {
    fmt::print(stderr, "GLFW error: {} ({})\n", error, desc);
}

}

void GlfwWindowResizeFunc(GLFWwindow *window, int width, int height) {
    auto *app = static_cast<RestirApp *>(glfwGetWindowUserPointer(window));
    app->Resize(width, height);
}

void GlfwCursorPosFunc(GLFWwindow *window, double x, double y) {
    auto *app = static_cast<RestirApp *>(glfwGetWindowUserPointer(window));
    uint8_t state = 0;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        state |= 1;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        state |= 2;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
        state |= 4;
    }
    app->OnMouse(x, y, state);
}

RestirApp::RestirApp(const RestirAppConfig &config) {
    InitializeBasic(config);
    InitializeGl();
    InitializeOptix();
    InitializeScene(config);
    InitializeAccel();
    InitializeSbt();
}

RestirApp::~RestirApp() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window_);
    glfwTerminate();
}

void RestirApp::InitializeBasic(const RestirAppConfig &config) {
    // glfw, gl
    glfwSetErrorCallback(GlfwErrorLogFunc);
    glfwInit();

    window_width_ = config.width;
    window_height_ = config.height;
    window_ = glfwCreateWindow(window_width_, window_height_, config.title.c_str(), nullptr, nullptr);
    if (window_ == nullptr) {
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window_);

    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, GlfwWindowResizeFunc);
    glfwSetCursorPosCallback(window_, GlfwCursorPosFunc);
    
    int gl_version = gladLoadGL(glfwGetProcAddress);
    if (gl_version == 0) {
        throw std::runtime_error("Failed to load OpenGL");
    }
    const int gl_major = GLAD_VERSION_MAJOR(gl_version);
    const int gl_minor = GLAD_VERSION_MINOR(gl_version);
    if (gl_major < 4 || gl_minor < 5) {
        throw std::runtime_error(fmt::format("OpenGL 4.5 is needed but {}.{} is loaded", gl_major, gl_minor));
    }

    // imgui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void) io;
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    // cuda, optix
    cudaFree(nullptr);
    int num_cuda_devices = 0;
    cudaGetDeviceCount(&num_cuda_devices);
    if (num_cuda_devices == 0) {
        throw std::runtime_error("Failed to find a CUDA device");
    }
    OPTIX_CHECK(optixInit());
    const int selected_device = 0;
    CUDA_CHECK(cudaSetDevice(selected_device));
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, selected_device);
    fmt::print("Selected device: {}\n", device_prop.name);
    CU_CHECK(cuCtxGetCurrent(&cuda_context_));
    OPTIX_CHECK(optixDeviceContextCreate(cuda_context_, nullptr, &optix_context_));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optix_context_, OptixLogFunc, nullptr, 4));

    optix_launch_params_buffer_.Alloc(sizeof(LaunchParams));

    // color buffer
    color_buffer_.Init(sizeof(pcm::Vec4) * window_width_ * window_height_, cudaGraphicsMapFlagsWriteDiscard);
    color_buffer_tex_.Create(window_width_, window_height_, GL_RGBA32F);
    
    optix_launch_params_.frame.width = window_width_;
    optix_launch_params_.frame.height = window_height_;

    // restir
    restir_config_.num_initial_samples = 32;
    restir_config_.num_eveluated_samples = 1;
    restir_config_.num_spatial_samples = 5;
    restir_config_.num_spatial_reuse_pass = 2;
    restir_config_.spatial_radius = 30;
    restir_config_.visibility_reuse = true;
    restir_config_.temporal_reuse = true;
    restir_config_.unbiased = false;
    restir_config_.mis_spatial_reuse = false;
    optix_launch_params_.restir.config = restir_config_;

    const size_t reserviors_buffer_size = sizeof(Reservoir) * restir_config_.num_eveluated_samples
        * window_width_ * window_height_;
    reservoirs_buffer_[0].Alloc(reserviors_buffer_size);
    reservoirs_buffer_[1].Alloc(reserviors_buffer_size);

    optix_launch_params_.light_strength_scale = config.light_strength_scale;
}

void RestirApp::InitializeGl() {
    // gbuffer pass
    gbuffer_shader_ = std::make_unique<GlProgram>(
        fmt::format("{}/shaders/gbuffer.vert", kProjectSourceDir),
        fmt::format("{}/shaders/gbuffer.frag", kProjectSourceDir)
    );
    gbuffer_fb_ = std::make_unique<GlFramebuffer>();
    gbuffer_fb_->CreateAsGBuffer(window_width_, window_height_);

    glCreateVertexArrays(1, &gbuffer_vao_);
    glEnableVertexArrayAttrib(gbuffer_vao_, 0);
    glVertexArrayAttribFormat(gbuffer_vao_, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(gbuffer_vao_, 0, 0);
    glEnableVertexArrayAttrib(gbuffer_vao_, 1);
    glVertexArrayAttribFormat(gbuffer_vao_, 1, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(gbuffer_vao_, 1, 1);
    glEnableVertexArrayAttrib(gbuffer_vao_, 2);
    glVertexArrayAttribFormat(gbuffer_vao_, 2, 4, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(gbuffer_vao_, 2, 2);
    glEnableVertexArrayAttrib(gbuffer_vao_, 3);
    glVertexArrayAttribFormat(gbuffer_vao_, 3, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(gbuffer_vao_, 3, 3);

    const size_t interop_buffer_size = sizeof(pcm::Vec4) * window_width_ * window_height_;
    gbuffer_base_color_emissive_.Init(interop_buffer_size, cudaGraphicsRegisterFlagsReadOnly);
    gbuffer_pos_roughness_.Init(interop_buffer_size, cudaGraphicsRegisterFlagsReadOnly);
    gbuffer_norm_metallic_.Init(interop_buffer_size, cudaGraphicsRegisterFlagsReadOnly);
    gbuffer_id_.Init(sizeof(uint32_t) * window_width_ * window_height_, cudaGraphicsRegisterFlagsReadOnly);
    gbuffer_prev_id_.Init(sizeof(uint32_t) * window_width_ * window_height_, cudaGraphicsRegisterFlagsReadOnly);

    // screen blit pass
    screen_blit_shader_ = std::make_unique<GlProgram>(
        fmt::format("{}/shaders/screen.vert", kProjectSourceDir),
        fmt::format("{}/shaders/screen.frag", kProjectSourceDir)
    );
    glCreateVertexArrays(1, &screen_blit_vao_);

    // state
    glEnable(GL_DEPTH_TEST);
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    glDepthFunc(GL_GREATER);
    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CW);
    glClearDepth(0.0);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glViewport(0, 0, window_width_, window_height_);
}

void RestirApp::InitializeOptix() {
    const OptixProgramConfig sample_config {
        .ptx_path = fmt::format("{}/raytracing/sample.ptx", kProjectSourceDir),
        .launch_params = std::string("optix_launch_params"),
        .max_trace_depth = 2,
        .raygen = std::string("__raygen__Sample"),
        .miss = {std::string("__miss__Shadow")},
        .closesthit = {std::string("__closesthit__Empty")},
        .anyhit = {std::string("__anyhit__Empty")},
    };
    program_sample_ = std::make_unique<OptixProgram>(sample_config, optix_context_);
    
    const OptixProgramConfig spatial_config {
        .ptx_path = fmt::format("{}/raytracing/spatial.ptx", kProjectSourceDir),
        .launch_params = std::string("optix_launch_params"),
        .max_trace_depth = 2,
        .raygen = std::string("__raygen__Spatial"),
        .miss = {std::string("__miss__Shadow")},
        .closesthit = {std::string("__closesthit__Empty")},
        .anyhit = {std::string("__anyhit__Empty")},
    };
    program_spatial_ = std::make_unique<OptixProgram>(spatial_config, optix_context_);
    
    const OptixProgramConfig lighting_config {
        .ptx_path = fmt::format("{}/raytracing/lighting.ptx", kProjectSourceDir),
        .launch_params = std::string("optix_launch_params"),
        .max_trace_depth = 2,
        .raygen = std::string("__raygen__Lighting"),
        .miss = {std::string("__miss__Shadow")},
        .closesthit = {std::string("__closesthit__Empty")},
        .anyhit = {std::string("__anyhit__Empty")},
    };
    program_lighting_ = std::make_unique<OptixProgram>(lighting_config, optix_context_);
}

void RestirApp::InitializeScene(const RestirAppConfig &config) {
    scene_ = std::make_unique<Scene>(config.scene_path);
    const size_t num_drawables = scene_->Drawables().size();

    // gl vb & ib
    const uint32_t num_vertices = scene_->Positions().size();
    glCreateBuffers(1, &scene_gl_vb_pos_);
    glNamedBufferStorage(scene_gl_vb_pos_, num_vertices * sizeof(pcm::Vec3), scene_->Positions().data(), 0);
    glCreateBuffers(1, &scene_gl_vb_norm_);
    glNamedBufferStorage(scene_gl_vb_norm_, num_vertices * sizeof(pcm::Vec3), scene_->Normals().data(), 0);
    glCreateBuffers(1, &scene_gl_vb_tan_);
    glNamedBufferStorage(scene_gl_vb_tan_, num_vertices * sizeof(pcm::Vec4), scene_->Tangents().data(), 0);
    glCreateBuffers(1, &scene_gl_vb_uv_);
    glNamedBufferStorage(scene_gl_vb_uv_, num_vertices * sizeof(pcm::Vec2), scene_->Uvs().data(), 0);

    glCreateBuffers(1, &scene_gl_ib_);
    glNamedBufferStorage(scene_gl_ib_, scene_->Indices().size() * sizeof(uint32_t), scene_->Indices().data(), 0);

    // gl drawable uniforms
    drawable_uniforms_.reserve(num_drawables);
    drawable_uniforms_buffer_.reserve(num_drawables);
    uint32_t drawable_id = 0;
    for (const auto &drawable : scene_->Drawables()) {
        ++drawable_id;
        DrawableUniforms uniforms {
            .model = drawable.model,
            .model_it = drawable.model.Inverse().Transpose(),
            .id = drawable_id,
        };

        GLuint id;
        glCreateBuffers(1, &id);
        glNamedBufferStorage(id, sizeof(DrawableUniforms), &uniforms, 0);

        drawable_uniforms_.emplace_back(uniforms);
        drawable_uniforms_buffer_.emplace_back(id);
    }

    // gl texs
    scene_textures_.reserve(scene_->Textures().size());
    for (const auto &tex : scene_->Textures()) {
        const auto &img = scene_->Images()[tex.source];
        const int levels = std::ceil(std::log2(std::max(img.width, img.height)));

        GLuint id;
        glCreateTextures(GL_TEXTURE_2D, 1, &id);
        glTextureStorage2D(id, levels, GL_RGBA8, img.width, img.height);
        glTextureSubImage2D(id, 0, 0, 0, img.width, img.height, GL_RGBA, GL_UNSIGNED_BYTE, img.image.data());
        glGenerateTextureMipmap(id);
        if (tex.sampler >= 0) {
            const auto &sampler = scene_->Samplers()[tex.sampler];
            glTextureParameteri(id, GL_TEXTURE_WRAP_S, sampler.wrapS);
            glTextureParameteri(id, GL_TEXTURE_WRAP_T, sampler.wrapT);
            if (sampler.minFilter != -1) {
                glTextureParameteri(id, GL_TEXTURE_MIN_FILTER, sampler.minFilter);
            }
            if (sampler.magFilter != -1) {
                glTextureParameteri(id, GL_TEXTURE_MIN_FILTER, sampler.magFilter);
            }
        }
        scene_textures_.push_back(id);
    }
    const uint8_t default_tex_white_data[] = { 255, 255, 255, 255 };
    glCreateTextures(GL_TEXTURE_2D, 1, &default_tex_white_);
    glTextureStorage2D(default_tex_white_, 1, GL_RGBA8, 1, 1);
    glTextureSubImage2D(default_tex_white_, 0, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, default_tex_white_data);
    const uint8_t default_tex_norm_data[] = { 127, 127, 255, 255 };
    glCreateTextures(GL_TEXTURE_2D, 1, &default_tex_norm_);
    glTextureStorage2D(default_tex_norm_, 1, GL_RGBA8, 1, 1);
    glTextureSubImage2D(default_tex_norm_, 0, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, default_tex_norm_data);

    // gl material uniforms
    const size_t num_materials = scene_->Materials().size();
    material_uniforms_.reserve(num_materials);
    material_uniforms_buffer_.reserve(num_materials);
    for (auto &material : scene_->Materials()) {
        MaterialUniforms uniforms {
            .base_color = material.base_color,
            .emissive = material.emissive,
            .metallic = material.metallic,
            .roughness = material.roughness,
            .alpha_mode = material.alpha_mode,
            .alpha_cutoff = material.alpha_cutoff,
            .normal_tex_scale = material.normal_tex_scale,
        };

        GLuint id;
        glCreateBuffers(1, &id);
        glNamedBufferStorage(id, sizeof(MaterialUniforms), &uniforms, 0);

        material_uniforms_.emplace_back(uniforms);
        material_uniforms_buffer_.emplace_back(id);
    }

    // camera
    camera_ = std::make_unique<Camera>(
        0.001f,
        pcm::Radians(45.0f),
        static_cast<float>(window_width_) / window_height_,
        std::numbers::pi * 0.25f,
        0.0f,
        (scene_->BoundingMax() - scene_->Center()).Length() * 2.5f,
        scene_->Center()
    );
    camera_uniforms_.proj_view = camera_->ProjViewMatrix();
    camera_uniforms_dirty_ = false;
    glCreateBuffers(1, &camera_uniforms_buffer_);
    glNamedBufferStorage(camera_uniforms_buffer_, sizeof(CameraUniforms), &camera_uniforms_, GL_MAP_WRITE_BIT);
    optix_launch_params_.camera.proj_view = camera_->ProjViewMatrix();
    optix_launch_params_.camera.position = camera_->Position();

    // lights
    scene_lights_ = std::make_unique<Lights>(scene_.get());
    scene_lights_buffer_.AllocAndUpload(scene_lights_->LightsData().data(),
        scene_lights_->LightsData().size() * sizeof(LightData));
    optix_launch_params_.light.data = scene_lights_buffer_.TypedPtr<LightData>();
    optix_launch_params_.light.light_count = scene_lights_->LightsData().size();
}

void RestirApp::InitializeAccel() {
    const size_t num_drawables = scene_->Drawables().size();

    scene_position_buffer_.AllocAndUpload(scene_->Positions().data(), scene_->Positions().size() * sizeof(pcm::Vec3));
    scene_normal_buffer_.AllocAndUpload(scene_->Normals().data(), scene_->Normals().size() * sizeof(pcm::Vec3));
    scene_index_buffer_.AllocAndUpload(scene_->Indices().data(), scene_->Indices().size() * sizeof(uint32_t));

    std::vector<float> drawables_model(num_drawables * 12);
    for (size_t i = 0, j = 0; i < num_drawables; i++) {
        const pcm::Mat4 &model = scene_->Drawables()[i].model;
        drawables_model[j++] = model[0][0];
        drawables_model[j++] = model[1][0];
        drawables_model[j++] = model[2][0];
        drawables_model[j++] = model[3][0];
        drawables_model[j++] = model[0][1];
        drawables_model[j++] = model[1][1];
        drawables_model[j++] = model[2][1];
        drawables_model[j++] = model[3][1];
        drawables_model[j++] = model[0][2];
        drawables_model[j++] = model[1][2];
        drawables_model[j++] = model[2][2];
        drawables_model[j++] = model[3][2];
    }
    scene_drawables_model_buffer_.AllocAndUpload(drawables_model.data(), drawables_model.size() * sizeof(float));

    const uint32_t input_flags[] = { 0 };
    std::vector<CUdeviceptr> vertex_buffer_ptrs(num_drawables);
    std::vector<OptixBuildInput> inputs(num_drawables);
    for (size_t i = 0; i < num_drawables; i++) {
        const auto &mesh = scene_->Meshes()[scene_->Drawables()[i].mesh_index];
        vertex_buffer_ptrs[i] = scene_position_buffer_.DevicePtr() + mesh.vertex_offset * sizeof(pcm::Vec3);
        
        inputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        inputs[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        inputs[i].triangleArray.vertexStrideInBytes = sizeof(pcm::Vec3);
        inputs[i].triangleArray.vertexBuffers = &vertex_buffer_ptrs[i];
        inputs[i].triangleArray.numVertices = mesh.num_vertices;

        inputs[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        inputs[i].triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
        inputs[i].triangleArray.indexBuffer = scene_index_buffer_.DevicePtr() + mesh.first_index * sizeof(uint32_t);
        inputs[i].triangleArray.numIndexTriplets = mesh.num_indices / 3;

        inputs[i].triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
        inputs[i].triangleArray.preTransform = scene_drawables_model_buffer_.DevicePtr() + i * 12 * sizeof(float);

        inputs[i].triangleArray.flags = input_flags;
        inputs[i].triangleArray.numSbtRecords = 1;
        inputs[i].triangleArray.sbtIndexOffsetBuffer = 0;
        inputs[i].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        inputs[i].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }

    OptixAccelBuildOptions build_options = {};
    build_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    build_options.motionOptions.numKeys = 1;
    build_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optix_context_,
        &build_options,
        inputs.data(),
        inputs.size(),
        &buffer_sizes
    ));

    CudaBuffer compacted_size_buffer;
    compacted_size_buffer.Alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emit_desc = {};
    emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = compacted_size_buffer.DevicePtr();

    CudaBuffer temp_buffer;
    temp_buffer.Alloc(buffer_sizes.tempSizeInBytes);
    CudaBuffer output_buffer;
    output_buffer.Alloc(buffer_sizes.outputSizeInBytes);
    OPTIX_CHECK(optixAccelBuild(
        optix_context_,
        nullptr,
        &build_options,
        inputs.data(),
        inputs.size(),
        temp_buffer.DevicePtr(),
        temp_buffer.Size(),
        output_buffer.DevicePtr(),
        output_buffer.Size(),
        &scene_accel_handle_,
        &emit_desc,
        1
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    uint64_t compacted_size;
    compacted_size_buffer.Download(&compacted_size, sizeof(uint64_t));

    scene_accel_buffer_.Alloc(compacted_size);
    OPTIX_CHECK(optixAccelCompact(
        optix_context_,
        nullptr,
        scene_accel_handle_,
        scene_accel_buffer_.DevicePtr(),
        scene_accel_buffer_.Size(),
        &scene_accel_handle_
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    output_buffer.Free();
    temp_buffer.Free();
    compacted_size_buffer.Free();

    // launch params
    optix_launch_params_.scene.indices = scene_index_buffer_.TypedPtr<uint32_t>();
    optix_launch_params_.scene.positions = scene_position_buffer_.TypedPtr<pcm::Vec3>();
    optix_launch_params_.scene.normals = scene_normal_buffer_.TypedPtr<pcm::Vec3>();
    optix_launch_params_.scene.traversable = scene_accel_handle_;
}

void RestirApp::InitializeSbt() {
    std::vector<std::nullptr_t> empty_hitgroup_data(scene_->Drawables().size());

    sbt_sample_ = std::make_unique<OptixSbt>(program_sample_.get(), empty_hitgroup_data);
    sbt_spatial_ = std::make_unique<OptixSbt>(program_spatial_.get(), empty_hitgroup_data);
    sbt_lighting_ = std::make_unique<OptixSbt>(program_lighting_.get(), empty_hitgroup_data);
}

void RestirApp::MainLoop() {
    frame_counter_.Reset();

    reservoirs_buffer_curr_index_ = 0;
    reservoirs_buffer_prev_valid_ = false;

    bool imgui_frame_stats_active = true;
    bool imgui_restir_config_active = true;

    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();

        frame_counter_.Tick();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        Update();
        Render();
        AfterRender();

        if (imgui_frame_stats_active) {
            ImGui::Begin("Frame Stats", &imgui_frame_stats_active);
            ImGui::Text("FPS: %.3f", frame_counter_.Fps());
            ImGui::Text("Per Frame: %.3f ms", frame_counter_.Mspf());
            ImGui::End();
        }

        if (imgui_restir_config_active) {
            ImGui::Begin("ReSTIR Config", &imgui_restir_config_active);
            ImguiConfigRestir();
            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window_, 1);
        }

        glfwSwapBuffers(window_);
    }
}

void RestirApp::Update() {
    optix_launch_params_.camera.prev_proj_view = optix_launch_params_.camera.proj_view;

    if (camera_uniforms_dirty_) {
        auto *ptr = static_cast<CameraUniforms *>(
            glMapNamedBufferRange(camera_uniforms_buffer_, 0, sizeof(CameraUniforms), GL_MAP_WRITE_BIT));
        *ptr = camera_uniforms_;
        glUnmapNamedBuffer(camera_uniforms_buffer_);
        camera_uniforms_dirty_ = false;
        glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
        
        optix_launch_params_.camera.proj_view = camera_->ProjViewMatrix();
        optix_launch_params_.camera.position = camera_->Position();
    }

    optix_launch_params_.frame.curr_time = frame_counter_.TotalTime() * 1000.0;
}

void RestirApp::Render() {
    // gbuffer
    glUseProgram(gbuffer_shader_->id);
    glBindVertexArray(gbuffer_vao_);
    glBindFramebuffer(GL_FRAMEBUFFER, gbuffer_fb_->id);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene_gl_ib_);
    glBindVertexBuffer(0, scene_gl_vb_pos_, 0, sizeof(pcm::Vec3));
    glBindVertexBuffer(1, scene_gl_vb_norm_, 0, sizeof(pcm::Vec3));
    glBindVertexBuffer(2, scene_gl_vb_tan_, 0, sizeof(pcm::Vec4));
    glBindVertexBuffer(3, scene_gl_vb_uv_, 0, sizeof(pcm::Vec2));

    glBindBufferRange(GL_UNIFORM_BUFFER, 0, camera_uniforms_buffer_, 0, sizeof(CameraUniforms));
    for (size_t i = 0; i < scene_->Drawables().size(); i++) {
        const auto &drawable = scene_->Drawables()[i];
        const auto &mesh = scene_->Meshes()[drawable.mesh_index];
        const auto &material = scene_->Materials()[mesh.material_index];

        if (material.double_sided) {
            glDisable(GL_CULL_FACE);
        } else {
            glEnable(GL_CULL_FACE);
        }

        glBindBufferRange(GL_UNIFORM_BUFFER, 1, drawable_uniforms_buffer_[i], 0, sizeof(DrawableUniforms));
        glBindBufferRange(GL_UNIFORM_BUFFER, 2, material_uniforms_buffer_[mesh.material_index],
            0, sizeof(MaterialUniforms));
        if (material.base_color_tex >= 0) {
            glBindTextureUnit(3, scene_textures_[material.base_color_tex]);
        } else {
            glBindTextureUnit(3, default_tex_white_);
        }
        if (material.emissive_tex >= 0) {
            glBindTextureUnit(4, scene_textures_[material.emissive_tex]);
        } else {
            glBindTextureUnit(4, default_tex_white_);
        }
        if (material.metallic_roughness_tex >= 0) {
            glBindTextureUnit(5, scene_textures_[material.metallic_roughness_tex]);
        } else {
            glBindTextureUnit(5, default_tex_white_);
        }
        if (material.normal_tex >= 0) {
            glBindTextureUnit(6, scene_textures_[material.normal_tex]);
        } else {
            glBindTextureUnit(6, default_tex_norm_);
        }

        glDrawElementsBaseVertex(
            GL_TRIANGLES,
            mesh.num_indices,
            GL_UNSIGNED_INT,
            (void *) (mesh.first_index * sizeof(uint32_t)),
            mesh.vertex_offset
        );
    }

    gbuffer_base_color_emissive_.PackFrom(*gbuffer_fb_, GL_COLOR_ATTACHMENT0);
    gbuffer_pos_roughness_.PackFrom(*gbuffer_fb_, GL_COLOR_ATTACHMENT1);
    gbuffer_norm_metallic_.PackFrom(*gbuffer_fb_, GL_COLOR_ATTACHMENT2);
    gbuffer_id_.PackFrom(*gbuffer_fb_, GL_COLOR_ATTACHMENT3);

    // sample
    optix_launch_params_.frame.color_buffer = nullptr;
    optix_launch_params_.frame.albedo_emissive_buffer = gbuffer_base_color_emissive_.TypedMap<pcm::Vec4>();
    optix_launch_params_.frame.pos_roughness_buffer = gbuffer_pos_roughness_.TypedMap<pcm::Vec4>();
    optix_launch_params_.frame.norm_metallic_buffer = gbuffer_norm_metallic_.TypedMap<pcm::Vec4>();
    optix_launch_params_.frame.id_buffer = gbuffer_id_.TypedMap<uint32_t>();
    optix_launch_params_.frame.prev_id_buffer = gbuffer_prev_id_.TypedMap<uint32_t>();
    optix_launch_params_.restir.reservoirs = reservoirs_buffer_[reservoirs_buffer_curr_index_].TypedPtr<Reservoir>();
    if (reservoirs_buffer_prev_valid_) {
        optix_launch_params_.restir.prev_reservoirs =
            reservoirs_buffer_[reservoirs_buffer_curr_index_ ^ 1].TypedPtr<Reservoir>();
    } else {
        optix_launch_params_.restir.prev_reservoirs = nullptr;
        reservoirs_buffer_prev_valid_ = true;
    }
    optix_launch_params_buffer_.Upload(&optix_launch_params_, sizeof(LaunchParams));
    reservoirs_buffer_curr_index_ ^= 1;

    OPTIX_CHECK(optixLaunch(
        program_sample_->Pipeline(),
        cuda_stream_,
        optix_launch_params_buffer_.DevicePtr(),
        optix_launch_params_buffer_.Size(),
        sbt_sample_->SbtPtr(),
        window_width_,
        window_height_,
        1
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // spatial
    for (uint8_t i = 0; i < restir_config_.num_spatial_reuse_pass; i++) {
        optix_launch_params_.restir.reservoirs =
            reservoirs_buffer_[reservoirs_buffer_curr_index_].TypedPtr<Reservoir>();
        optix_launch_params_.restir.prev_reservoirs =
            reservoirs_buffer_[reservoirs_buffer_curr_index_ ^ 1].TypedPtr<Reservoir>();
        optix_launch_params_buffer_.Upload(&optix_launch_params_, sizeof(LaunchParams));
        reservoirs_buffer_curr_index_ ^= 1;

        OPTIX_CHECK(optixLaunch(
            program_spatial_->Pipeline(),
            cuda_stream_,
            optix_launch_params_buffer_.DevicePtr(),
            optix_launch_params_buffer_.Size(),
            sbt_spatial_->SbtPtr(),
            window_width_,
            window_height_,
            1
        ));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // lighting
    optix_launch_params_.frame.color_buffer = color_buffer_.TypedMap<pcm::Vec4>();
    optix_launch_params_buffer_.Upload(&optix_launch_params_, sizeof(LaunchParams));

    OPTIX_CHECK(optixLaunch(
        program_lighting_->Pipeline(),
        cuda_stream_,
        optix_launch_params_buffer_.DevicePtr(),
        optix_launch_params_buffer_.Size(),
        sbt_lighting_->SbtPtr(),
        window_width_,
        window_height_,
        1
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // screen blit
    color_buffer_.UnpackTo(color_buffer_tex_);

    glUseProgram(screen_blit_shader_->id);
    glBindVertexArray(screen_blit_vao_);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindVertexBuffer(0, 0, 0, 0);
    glBindTextureUnit(0, color_buffer_tex_.id);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

void RestirApp::AfterRender() {
    gbuffer_base_color_emissive_.Unmap();
    gbuffer_pos_roughness_.Unmap();
    gbuffer_norm_metallic_.Unmap();
    gbuffer_id_.Unmap();
    gbuffer_prev_id_.Unmap();

    color_buffer_.Unmap();

    gbuffer_id_.SwapWith(gbuffer_prev_id_);
}

void RestirApp::ImguiConfigRestir() {
    int temp_num_initial_samples = restir_config_.num_initial_samples;
    ImGui::SliderInt("# initial samples", &temp_num_initial_samples, 1, 64);
    restir_config_.num_initial_samples = temp_num_initial_samples;

    int temp_num_evaluated_samples = restir_config_.num_eveluated_samples;
    ImGui::SliderInt("# evaluated samples", &temp_num_evaluated_samples, 1, 4);
    if (restir_config_.num_eveluated_samples != temp_num_evaluated_samples) {
        restir_config_.num_eveluated_samples = temp_num_evaluated_samples;
        const size_t reserviors_buffer_size = sizeof(Reservoir) * restir_config_.num_eveluated_samples
            * window_width_ * window_height_;
        reservoirs_buffer_[0].Resize(reserviors_buffer_size);
        reservoirs_buffer_[1].Resize(reserviors_buffer_size);
        reservoirs_buffer_prev_valid_ = false;
    }

    int temp_num_spatial_reuse_pass = restir_config_.num_spatial_reuse_pass;
    ImGui::SliderInt("# spatial reuse pass", &temp_num_spatial_reuse_pass, 0, 3);
    restir_config_.num_spatial_reuse_pass = temp_num_spatial_reuse_pass;

    int temp_num_spatial_samples = restir_config_.num_spatial_samples;
    ImGui::SliderInt("# spatial samples", &temp_num_spatial_samples, 1, 8);
    restir_config_.num_spatial_samples = temp_num_spatial_samples;

    int temp_spatial_radius = restir_config_.spatial_radius;
    ImGui::SliderInt("# spatial radius", &temp_spatial_radius, 0, 64);
    restir_config_.spatial_radius = temp_spatial_radius;

    ImGui::Checkbox("visibility reuse", &restir_config_.visibility_reuse);

    ImGui::Checkbox("temporal reuse", &restir_config_.temporal_reuse);

    ImGui::Checkbox("unbiased spatial reuse", &restir_config_.unbiased);

    ImGui::Checkbox("mis spatial reuse", &restir_config_.mis_spatial_reuse);

    optix_launch_params_.restir.config = restir_config_;
}

void RestirApp::Resize(int width, int height) {
    if (width == 0 || height == 0 || (width == window_width_ && height == window_height_)) {
        return;
    }

    window_width_ = width;
    window_height_ = height;

    optix_launch_params_.frame.width = width;
    optix_launch_params_.frame.height = height;

    const size_t interop_buffer_size = sizeof(pcm::Vec4) * width * height;

    color_buffer_.Resize(interop_buffer_size, cudaGraphicsMapFlagsWriteDiscard);
    color_buffer_tex_.Delete();
    color_buffer_tex_.Create(window_width_, window_height_, GL_RGBA32F);

    gbuffer_base_color_emissive_.Resize(interop_buffer_size, cudaGraphicsRegisterFlagsReadOnly);
    gbuffer_pos_roughness_.Resize(interop_buffer_size, cudaGraphicsRegisterFlagsReadOnly);
    gbuffer_norm_metallic_.Resize(interop_buffer_size, cudaGraphicsRegisterFlagsReadOnly);
    gbuffer_id_.Resize(sizeof(uint32_t) * width * height, cudaGraphicsRegisterFlagsReadOnly);
    gbuffer_prev_id_.Resize(sizeof(uint32_t) * width * height, cudaGraphicsRegisterFlagsReadOnly);
    
    glViewport(0, 0, width, height);
    
    const size_t reserviors_buffer_size = sizeof(Reservoir) * restir_config_.num_eveluated_samples * width * height;
    reservoirs_buffer_[0].Resize(reserviors_buffer_size);
    reservoirs_buffer_[1].Resize(reserviors_buffer_size);
    reservoirs_buffer_prev_valid_ = false;

    if (gbuffer_fb_) {
        gbuffer_fb_->Delete();
        gbuffer_fb_->CreateAsGBuffer(width, height);
    }

    if (camera_) {
        camera_->UpdateAspect(static_cast<float>(width) / height);
        camera_uniforms_.proj_view = camera_->ProjViewMatrix();
        camera_uniforms_dirty_ = true;
    }
}

void RestirApp::OnMouse(double x, double y, uint8_t state) {
    if (state & 1) {
        const float dx = 0.25 * (x - last_mouse_x_) * std::numbers::pi / 180.0;
        const float dy = 0.25 * (y - last_mouse_y_) * std::numbers::pi / 180.0;
        camera_->Rotate(dy, -dx);
    } else if (state & 2) {
        const float dx = 0.005 * (x - last_mouse_x_);
        const float dy = 0.005 * (y - last_mouse_y_);
        camera_->Stretch((dx - dy) * 50.0f);
    }
    last_mouse_x_ = x;
    last_mouse_y_ = y;

    if (state & 3) {
        camera_uniforms_.proj_view = camera_->ProjViewMatrix();
        camera_uniforms_dirty_ = true;
    }
}
# glfw
find_package(fmt CONFIG REQUIRED)

# glfw
find_package(glfw3 CONFIG REQUIRED)

# imgui
find_package(imgui CONFIG REQUIRED)

# cuda toolkit
find_package(CUDAToolkit REQUIRED)

# cuda toolkit
find_package(OptiX REQUIRED)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/restir_optix.hpp.in
    ${CMAKE_CURRENT_SOURCE_DIR}/restir_optix.hpp
)

# target

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/restir_optix.hpp
)

target_include_directories(${PROJECT_NAME}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        OptiX::OptiX
        CUDA::cudart CUDA::cuda_driver
        pep-cuda-math
        fmt::fmt
        glfw
        glad2
        imgui::imgui
        tinygltf
        mikktspace
)

target_compile_features(${PROJECT_NAME}
    PRIVATE cxx_std_20
)

add_subdirectory(app)
add_subdirectory(cuda_utils)
add_subdirectory(gl_utils)
add_subdirectory(misc)
add_subdirectory(raytracing)
add_subdirectory(scene)
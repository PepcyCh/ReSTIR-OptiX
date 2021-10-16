add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/sample.ptx
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/sample.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/launch_params.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/raytracing_common.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/reservoir.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shading.hpp
    COMMAND nvcc ${CMAKE_CURRENT_SOURCE_DIR}/sample.cu
        -ptx -o ${CMAKE_CURRENT_SOURCE_DIR}/sample.ptx -std=c++17
        -I ${OptiX_ROOT_DIR}/include
        -I ${PROJECT_SOURCE_DIR}/src
        -I ${PROJECT_SOURCE_DIR}/external/pep-cuda-math/include -DPCMATH_NAMESPACE_ALIAS
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/spatial.ptx
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/spatial.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/launch_params.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/raytracing_common.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/reservoir.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shading.hpp
    COMMAND nvcc ${CMAKE_CURRENT_SOURCE_DIR}/spatial.cu
        -ptx -o ${CMAKE_CURRENT_SOURCE_DIR}/spatial.ptx -std=c++17
        -I ${OptiX_ROOT_DIR}/include
        -I ${PROJECT_SOURCE_DIR}/src
        -I ${PROJECT_SOURCE_DIR}/external/pep-cuda-math/include -DPCMATH_NAMESPACE_ALIAS
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/lighting.ptx
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/lighting.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/launch_params.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/raytracing_common.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/reservoir.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shading.hpp
    COMMAND nvcc ${CMAKE_CURRENT_SOURCE_DIR}/lighting.cu
        -ptx -o ${CMAKE_CURRENT_SOURCE_DIR}/lighting.ptx -std=c++17
        -I ${OptiX_ROOT_DIR}/include
        -I ${PROJECT_SOURCE_DIR}/src
        -I ${PROJECT_SOURCE_DIR}/external/pep-cuda-math/include -DPCMATH_NAMESPACE_ALIAS
)
add_custom_target(raytracing-ptx
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/sample.ptx
        ${CMAKE_CURRENT_SOURCE_DIR}/spatial.ptx
        ${CMAKE_CURRENT_SOURCE_DIR}/lighting.ptx
)

target_sources(${PROJECT_NAME}
    PRIVATE
        launch_params.hpp
        reservoir.hpp
        rng.hpp
        shading.hpp
)

add_dependencies(${PROJECT_NAME} raytracing-ptx)
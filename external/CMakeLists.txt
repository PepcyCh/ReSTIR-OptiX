# glad
add_library(glad2
    glad/src/gl.c
    glad/include/glad/gl.h
    glad/include/KHR/khrplatform.h
)
target_include_directories(glad2
    PUBLIC glad/include
)

# mikktspace
add_library(mikktspace
    mikktspace/mikktspace.c
    mikktspace/mikktspace.h
)
target_include_directories(mikktspace
    PUBLIC mikktspace
)

# tinygltf
set(TINYGLTF_HEADER_ONLY ON CACHE INTERNAL "" FORCE)
set(TINYGLTF_INSTALL_OFF OFF CACHE INTERNAL "" FORCE)
add_subdirectory(tinygltf)

# pep-cuda-math
set(PCMATH_BUILD_SAMPLES OFF CACHE INTERNAL "" FORCE)
add_subdirectory(pep-cuda-math)
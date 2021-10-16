if(DEFINED ENV{OptiX_INSTALL_DIR})
    message(STATUS "Detected the OptiX_INSTALL_DIR env variable (pointing to $ENV{OptiX_INSTALL_DIR}; going to use this for finding optix.h")
    find_path(OptiX_ROOT_DIR NAMES include/optix.h PATHS $ENV{OptiX_INSTALL_DIR})
elseif(WIN32)
    find_path(OptiX_ROOT_DIR
        NAMES include/optix.h
        PATHS
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.2.0"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.1.0"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0"
    )
else()
    find_path(OptiX_ROOT_DIR NAMES include/optix.h)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
    FAIL_MESSAGE "Failed to find OptiX install dir. Please instal OptiX or set OptiX_INSTALL_DIR env variable."
    REQUIRED_VARS OptiX_ROOT_DIR
)

add_library(OptiX::OptiX INTERFACE IMPORTED)
target_include_directories(OptiX::OptiX INTERFACE ${OptiX_ROOT_DIR}/include)
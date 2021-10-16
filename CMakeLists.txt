cmake_minimum_required(VERSION 3.18)

project(ReSTIR-OptiX)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(FATAL "In-source builds not allowed.")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_subdirectory(external)

add_subdirectory(src)
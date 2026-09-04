cmake_minimum_required(VERSION 3.24)

project(Renderer
    VERSION 0.1.0
    DESCRIPTION "Modular Windows Vulkan renderer"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Group engine libs and third-party targets into IDE folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Shared warning level for all engine/app targets (deliberately not global,
# so third-party code builds with its own settings)
add_library(rend_warnings INTERFACE)
if(MSVC)
    target_compile_options(rend_warnings INTERFACE /W4 /permissive-)
else()
    target_compile_options(rend_warnings INTERFACE -Wall -Wextra -Wpedantic)
endif()

include(cmake/FetchSDL3.cmake)
include(cmake/FetchVolk.cmake)

# One shared output dir so viewer.exe sits next to rend.dll.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(assets)
add_subdirectory(engine)
add_subdirectory(apps/viewer)

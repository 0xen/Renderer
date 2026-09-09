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
include(cmake/FetchPugixml.cmake)
include(cmake/FetchCgltf.cmake)
include(cmake/FetchStb.cmake)
include(cmake/FetchImgui.cmake)
include(cmake/FetchMeshoptimizer.cmake)

# One shared output dir so viewer.exe sits next to rend.dll.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Opt-in Tracy profiling: normal builds carry zero overhead (the zone
# macros in rend/core/profile.h compile away without REND_TRACY).
option(REND_TRACY "Build with the Tracy profiler client (CPU zones + frame marks)" OFF)
if(REND_TRACY)
    include(cmake/FetchTracy.cmake)
endif()

# The Python host: a standalone optional DLL (like assetio) that embeds
# CPython and drives the renderer message queue from scene scripts. The
# viewer loads it at RUNTIME only when a scene declares <Script> nodes,
# so turning this off (or deleting the DLL) costs nothing.
option(REND_PYTHON "Build the embedded Python script host DLL" ON)
if(REND_PYTHON)
    include(cmake/FetchPython.cmake)
    include(cmake/FetchPybind11.cmake)
endif()

add_subdirectory(assets)
add_subdirectory(engine)
add_subdirectory(assetio)
if(REND_PYTHON)
    add_subdirectory(pyhost)
endif()
add_subdirectory(apps/viewer)

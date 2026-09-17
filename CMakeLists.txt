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

# The engine DLL itself needs only the windowing backend and the Vulkan
# loader; the fetches under REND_BUILD_APPS serve assetio and the viewer.
include(cmake/FetchSDL3.cmake)
include(cmake/FetchVolk.cmake)
include(cmake/RendShaders.cmake)

# A project consuming the engine via add_subdirectory() wants rend.dll and
# the rend_add_shaders() rule only: turning this off skips the viewer,
# assetio, the engine's own shader target and the third-party fetches only
# those need. It no longer skips the Python host: REND_PYTHON stands on its
# own, below.
option(REND_BUILD_APPS "Build the viewer app, assetio and the engine shaders" ON)
if(REND_BUILD_APPS)
    include(cmake/FetchPugixml.cmake)
    include(cmake/FetchCgltf.cmake)
    include(cmake/FetchStb.cmake)
    include(cmake/FetchImgui.cmake)
    include(cmake/FetchMeshoptimizer.cmake)
endif()

# One shared output dir so viewer.exe sits next to rend.dll. A consuming
# project that set its own runtime directory keeps it: CMAKE_BINARY_DIR is
# then the consumer's build tree, so rend.dll lands next to its executable.
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
#
# This option alone decides it. It used to be gated on REND_BUILD_APPS as
# well, which made "standalone" above untrue in the one case that matters:
# a project consuming the engine with the apps off could not build the
# script host however loudly it asked for it. Nothing under pyhost/ needs
# the viewer, assetio or any fetch those two bring in — it links
# rend::engine and pybind11::embed and nothing else — so the second gate
# bought nothing and cost exactly that consumer.
option(REND_PYTHON "Build the embedded Python script host DLL" ON)
if(REND_PYTHON)
    include(cmake/FetchPython.cmake)
    include(cmake/FetchPybind11.cmake)
endif()

add_subdirectory(engine)
# Ahead of apps/viewer, so the DLL the viewer LoadLibrary's is built beside it.
if(REND_PYTHON)
    add_subdirectory(pyhost)
endif()
if(REND_BUILD_APPS)
    add_subdirectory(assets)
    add_subdirectory(assetio)
    add_subdirectory(apps/viewer)
endif()

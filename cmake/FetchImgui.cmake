# Fetches Dear ImGui (viewer debug UI), pinned. The repo ships no CMake
# project; build the core plus the Vulkan backend as a static lib. The
# backend uses volk for its function pointers (IMGUI_IMPL_VULKAN_USE_VOLK),
# so consumers call volkInitialize/volkLoadInstance themselves — the viewer
# has its own volk table, separate from the engine DLL's.
include(FetchContent)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.8
    GIT_SHALLOW ON
    SOURCE_SUBDIR does_not_exist_no_cmake_project
)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp)
target_include_directories(imgui SYSTEM PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK)
target_link_libraries(imgui PUBLIC volk)
set_target_properties(imgui PROPERTIES FOLDER "third_party")

# Fetches cgltf (single-header C glTF 2.0 parser used by the assetio glTF
# importer), pinned. The repo ships no CMake project; expose it as an
# interface target carrying only the include directory.
include(FetchContent)

FetchContent_Declare(
    cgltf
    GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
    GIT_TAG v1.15
    GIT_SHALLOW ON
    SOURCE_SUBDIR does_not_exist_no_cmake_project
)
FetchContent_MakeAvailable(cgltf)

add_library(cgltf INTERFACE)
target_include_directories(cgltf SYSTEM INTERFACE ${cgltf_SOURCE_DIR})

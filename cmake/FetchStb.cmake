# Fetches stb (stb_image decodes texture files in assetio). No releases
# upstream, so pinned to a commit hash instead of a tag; header-only, no
# CMake project of its own.
include(FetchContent)

FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG 2c980bb59875b0d32144a71867fbdebb2f77cd20
    SOURCE_SUBDIR does_not_exist_no_cmake_project
)
FetchContent_MakeAvailable(stb)

add_library(stb INTERFACE)
target_include_directories(stb SYSTEM INTERFACE ${stb_SOURCE_DIR})

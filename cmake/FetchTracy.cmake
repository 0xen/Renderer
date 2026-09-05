# Fetches the Tracy profiler client, pinned. Only included when REND_TRACY
# is ON (root CMakeLists), so normal builds never see it. Built SHARED:
# rend.dll and viewer.exe both emit zones, and they must feed one client
# instance — a static TracyClient in each module would run two.
include(FetchContent)

set(TRACY_ENABLE ON CACHE BOOL "" FORCE)
set(TRACY_STATIC OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG v0.14.1
    GIT_SHALLOW ON
)
FetchContent_MakeAvailable(tracy)

set_target_properties(TracyClient PROPERTIES FOLDER "third_party")

# Fetches meshoptimizer (zeux), pinned: the viewer uses meshopt_simplify /
# meshopt_simplifyScale at scene load to build the discrete LOD index
# chains the GPU cull pass picks from. Ships a normal CMake project that
# defines the `meshoptimizer` static library target.
include(FetchContent)

FetchContent_Declare(
    meshoptimizer
    GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
    GIT_TAG v0.22
    GIT_SHALLOW ON
)
FetchContent_MakeAvailable(meshoptimizer)
set_target_properties(meshoptimizer PROPERTIES FOLDER "third_party")

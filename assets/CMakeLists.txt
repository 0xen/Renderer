# Offline shader compilation. Single-source HLSL through dxc from the Vulkan
# SDK, which emits SPIR-V natively (and DXIL later for a D3D backend) — see
# docs/SCENE_FORMAT.md. There is no runtime shader compilation.
find_program(REND_DXC_EXECUTABLE
    NAMES dxc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    DOC "DirectX Shader Compiler (emits the SPIR-V artifacts for the Vulkan backend)")
if(NOT REND_DXC_EXECUTABLE)
    message(FATAL_ERROR "dxc not found; set VULKAN_SDK or REND_DXC_EXECUTABLE")
endif()
message(STATUS "Shader compiler: ${REND_DXC_EXECUTABLE}")

# Artifacts land next to the executable so the viewer resolves them relative
# to its own directory: <runtime dir>/data/shaders/<name>.<stage>.spv
set(REND_SHADER_OUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/data/shaders")

# source | entry point | dxc profile | artifact
set(REND_SHADER_STAGES
    "triangle.hlsl|VSMain|vs_6_0|triangle.vert.spv"
    "triangle.hlsl|PSMain|ps_6_0|triangle.frag.spv"
    "scene.hlsl|VSMain|vs_6_0|scene.vert.spv"
    "scene.hlsl|PSMain|ps_6_0|scene.frag.spv"
    "cull.hlsl|CSMain|cs_6_0|cull.comp.spv"
)

set(REND_SHADER_ARTIFACTS "")
foreach(stage IN LISTS REND_SHADER_STAGES)
    string(REPLACE "|" ";" parts "${stage}")
    list(GET parts 0 source)
    list(GET parts 1 entrypoint)
    list(GET parts 2 profile)
    list(GET parts 3 artifact)

    set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${source}")
    set(artifact_path "${REND_SHADER_OUT_DIR}/${artifact}")

    add_custom_command(
        OUTPUT "${artifact_path}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${REND_SHADER_OUT_DIR}"
        # -fspv-entrypoint-name renames the emitted entry point to "main" so
        # every stage looks the same to the pipeline definition format; it
        # must follow -E, which still names the HLSL function.
        COMMAND "${REND_DXC_EXECUTABLE}"
                -spirv -fspv-target-env=vulkan1.3
                -T ${profile} -E ${entrypoint} -fspv-entrypoint-name=main
                -Fo "${artifact_path}" "${source_path}"
        DEPENDS "${source_path}"
        COMMENT "dxc ${source} [${profile} ${entrypoint}] -> ${artifact}"
        VERBATIM)
    list(APPEND REND_SHADER_ARTIFACTS "${artifact_path}")
endforeach()

# The .hlsl sources are listed for IDE browsing only — Visual Studio would
# otherwise hand them to its built-in FXC rule instead of our dxc command.
set_source_files_properties(shaders/triangle.hlsl shaders/scene.hlsl shaders/cull.hlsl
    PROPERTIES VS_TOOL_OVERRIDE "None")

add_custom_target(shaders ALL DEPENDS ${REND_SHADER_ARTIFACTS}
    SOURCES shaders/triangle.hlsl shaders/scene.hlsl shaders/cull.hlsl)
set_target_properties(shaders PROPERTIES FOLDER "assets")

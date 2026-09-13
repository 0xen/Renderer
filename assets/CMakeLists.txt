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

# source | entry point | dxc profile | artifact | optional -D defines
set(REND_SHADER_STAGES
    "scene.hlsl|VSMain|vs_6_0|scene.vert.spv"
    "scene.hlsl|PSMain|ps_6_0|scene.frag.spv|-DRT_SHADOWS=0"
    # ps_6_5: RayQuery needs shader model 6.5 (SPV_KHR_ray_query).
    "scene.hlsl|PSMain|ps_6_5|scene_rt.frag.spv|-DRT_SHADOWS=1"
    "cull.hlsl|CSMain|cs_6_0|cull.comp.spv"
    "skin.hlsl|CSMain|cs_6_0|skin.comp.spv"
    "rt_primary.hlsl|VSMain|vs_6_0|rt_primary.vert.spv"
    # ps_6_5: RayQuery needs shader model 6.5 (SPV_KHR_ray_query).
    "rt_primary.hlsl|PSMain|ps_6_5|rt_primary.frag.spv"
    "shadow.hlsl|VSMain|vs_6_0|shadow.vert.spv"
    "shadow.hlsl|PSMain|ps_6_0|shadow.frag.spv"
    "proxy.hlsl|VSMain|vs_6_0|proxy.vert.spv"
    "proxy.hlsl|PSMain|ps_6_0|proxy.frag.spv"
    "proxy.hlsl|PSDebug|ps_6_0|proxy_debug.frag.spv"
    "proxy.hlsl|VSInstances|vs_6_0|proxy_instances.vert.spv"
    "obb.hlsl|CSMain|cs_6_0|obb.comp.spv"
    "sky.hlsl|VSMain|vs_6_0|sky.vert.spv"
    "sky.hlsl|PSMain|ps_6_0|sky.frag.spv"
    "point_shadow.hlsl|PSMain|ps_6_0|point_shadow.frag.spv"
    "gbuffer.hlsl|PSMain|ps_6_0|gbuffer.frag.spv"
    "deferred.hlsl|VSMain|vs_6_0|deferred.vert.spv"
    "deferred.hlsl|PSMain|ps_6_0|deferred.frag.spv|-DRT_SHADOWS=0"
    # ps_6_5: RayQuery needs shader model 6.5 (SPV_KHR_ray_query).
    "deferred.hlsl|PSMain|ps_6_5|deferred_rt.frag.spv|-DRT_SHADOWS=1"
    "post.hlsl|VSMain|vs_6_0|post.vert.spv"
    "post.hlsl|PSMain|ps_6_0|post.frag.spv"
)

set(REND_SHADER_ARTIFACTS "")
foreach(stage IN LISTS REND_SHADER_STAGES)
    string(REPLACE "|" ";" parts "${stage}")
    list(GET parts 0 source)
    list(GET parts 1 entrypoint)
    list(GET parts 2 profile)
    list(GET parts 3 artifact)
    set(defines "")
    list(LENGTH parts partCount)
    if(partCount GREATER 4)
        list(GET parts 4 defines)
    endif()

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
                ${defines}
                -Fo "${artifact_path}" "${source_path}"
        # Every stage depends on the shared includes; dxc resolves the
        # #include paths relative to the source file itself.
        DEPENDS "${source_path}"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shading.hlsli"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/lighting.hlsli"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rt_common.hlsli"
        COMMENT "dxc ${source} [${profile} ${entrypoint}] -> ${artifact}"
        VERBATIM)
    list(APPEND REND_SHADER_ARTIFACTS "${artifact_path}")
endforeach()

# The .hlsl sources are listed for IDE browsing only — Visual Studio would
# otherwise hand them to its built-in FXC rule instead of our dxc command.
set_source_files_properties(
    shaders/scene.hlsl shaders/cull.hlsl shaders/shadow.hlsl
    shaders/rt_primary.hlsl shaders/skin.hlsl shaders/proxy.hlsl shaders/obb.hlsl
    shaders/sky.hlsl
    shaders/point_shadow.hlsl shaders/gbuffer.hlsl shaders/deferred.hlsl
    shaders/post.hlsl
    shaders/shading.hlsli shaders/lighting.hlsli shaders/rt_common.hlsli
    PROPERTIES VS_TOOL_OVERRIDE "None")

add_custom_target(shaders ALL DEPENDS ${REND_SHADER_ARTIFACTS}
    SOURCES shaders/scene.hlsl shaders/cull.hlsl shaders/shadow.hlsl
            shaders/rt_primary.hlsl shaders/skin.hlsl shaders/proxy.hlsl shaders/obb.hlsl
            shaders/sky.hlsl
            shaders/point_shadow.hlsl shaders/gbuffer.hlsl shaders/deferred.hlsl
            shaders/post.hlsl
            shaders/shading.hlsli shaders/lighting.hlsli shaders/rt_common.hlsli)
set_target_properties(shaders PROPERTIES FOLDER "assets")

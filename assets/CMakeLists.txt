# The engine's own shaders, compiled by the shared dxc rule in
# cmake/RendShaders.cmake — see docs/SCENE_FORMAT.md. Artifacts land next to
# the executable so the viewer resolves them relative to its own directory:
# <runtime dir>/data/shaders/<name>.<stage>.spv
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
    "fxaa.hlsl|VSMain|vs_6_0|fxaa.vert.spv"
    "fxaa.hlsl|PSMain|ps_6_0|fxaa.frag.spv"
)

rend_add_shaders(shaders
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    OUTPUT_DIR "${REND_SHADER_OUT_DIR}"
    STAGES ${REND_SHADER_STAGES}
    INCLUDES shading.hlsli lighting.hlsli rt_common.hlsli)
set_target_properties(shaders PROPERTIES FOLDER "assets")

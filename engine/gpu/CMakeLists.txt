# Vulkan layer: instance, adapter selection/feature negotiation, device,
# swapchain, and the base GPU objects (shader, pipeline) plus the frame loop.
add_library(rend_gpu OBJECT
    src/acceleration_structure.cpp
    src/buffer.cpp
    src/command_context.cpp
    src/descriptor_table.cpp
    src/image.cpp
    src/instance.cpp
    src/device.cpp
    src/memory_pool.cpp
    src/memory_tracker.cpp
    src/swapchain.cpp
    src/shader.cpp
    src/pipeline.cpp
    src/probe_capture.cpp
    src/frame_renderer.cpp
    src/vulkan/vulkan_frame_renderer.cpp
    src/d3d12/d3d12_backend.cpp
    src/d3d12/d3d12_resources.cpp
    src/d3d12/d3d12_swapchain.cpp
    src/d3d12/d3d12_command_context.cpp
    src/d3d12/d3d12_descriptor_table.cpp
    src/d3d12/d3d12_pipeline.cpp
    src/d3d12/d3d12_upload.cpp
    src/d3d12/d3d12_frame_renderer.cpp
    src/texture_uploader.cpp
    src/transfer.cpp
)
add_library(rend::gpu ALIAS rend_gpu)

target_include_directories(rend_gpu PUBLIC include)
target_link_libraries(rend_gpu
    PUBLIC rend::core
    PRIVATE rend_warnings volk
)
if(WIN32)
    # Adapter selection asks DXGI which adapter owns the display; the D3D12
    # backend needs the runtime and DirectComposition (transparent targets).
    target_link_libraries(rend_gpu PRIVATE dxgi d3d12 dcomp)
endif()
set_target_properties(rend_gpu PROPERTIES FOLDER "engine")

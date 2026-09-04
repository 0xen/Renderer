# Vulkan layer: instance, adapter selection/feature negotiation, device,
# swapchain, and the base GPU objects (shader, pipeline) plus the frame loop.
add_library(rend_gpu OBJECT
    src/instance.cpp
    src/device.cpp
    src/swapchain.cpp
    src/shader.cpp
    src/pipeline.cpp
    src/frame_renderer.cpp
)
add_library(rend::gpu ALIAS rend_gpu)

target_include_directories(rend_gpu PUBLIC include)
target_link_libraries(rend_gpu
    PUBLIC rend::core
    PRIVATE rend_warnings volk
)
set_target_properties(rend_gpu PROPERTIES FOLDER "engine")

# Vulkan layer: instance, adapter selection/feature negotiation, device,
# swapchain, and the base GPU objects (shader, pipeline) plus the frame loop.
add_library(rend_gpu OBJECT
    src/buffer.cpp
    src/instance.cpp
    src/device.cpp
    src/mega_buffer.cpp
    src/swapchain.cpp
    src/shader.cpp
    src/pipeline.cpp
    src/frame_renderer.cpp
    src/transfer.cpp
)
add_library(rend::gpu ALIAS rend_gpu)

target_include_directories(rend_gpu PUBLIC include)
target_link_libraries(rend_gpu
    PUBLIC rend::core
    PRIVATE rend_warnings volk
)
if(WIN32)
    # Adapter selection asks DXGI which adapter owns the display.
    target_link_libraries(rend_gpu PRIVATE dxgi)
endif()
set_target_properties(rend_gpu PROPERTIES FOLDER "engine")

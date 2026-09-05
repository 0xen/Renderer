# Vulkan layer: instance, adapter selection/feature negotiation, device,
# swapchain, and the base GPU objects (shader, pipeline) plus the frame loop.
add_library(rend_gpu OBJECT
    src/acceleration_structure.cpp
    src/buffer.cpp
    src/descriptor_table.cpp
    src/image.cpp
    src/instance.cpp
    src/device.cpp
    src/memory_pool.cpp
    src/swapchain.cpp
    src/shader.cpp
    src/pipeline.cpp
    src/probe_capture.cpp
    src/frame_renderer.cpp
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
    # Adapter selection asks DXGI which adapter owns the display.
    target_link_libraries(rend_gpu PRIVATE dxgi)
endif()
set_target_properties(rend_gpu PROPERTIES FOLDER "engine")

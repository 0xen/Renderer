# Vulkan layer: instance today; adapter selection, device, swapchain and
# feature negotiation land in the coming milestones.
add_library(rend_gpu OBJECT
    src/instance.cpp
    src/device.cpp
)
add_library(rend::gpu ALIAS rend_gpu)

target_include_directories(rend_gpu PUBLIC include)
target_link_libraries(rend_gpu
    PUBLIC rend::core
    PRIVATE rend_warnings volk
)
set_target_properties(rend_gpu PROPERTIES FOLDER "engine")

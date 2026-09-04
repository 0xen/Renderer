# Placeholder: becomes the Vulkan layer (instance, adapter selection,
# device, swapchain, feature negotiation) starting next milestone.
add_library(rend_gpu INTERFACE)
add_library(rend::gpu ALIAS rend_gpu)
target_include_directories(rend_gpu INTERFACE include)
target_link_libraries(rend_gpu INTERFACE rend::core)

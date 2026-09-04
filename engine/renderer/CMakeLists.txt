# Placeholder: becomes the high-level rendering layer (frame graph,
# passes, materials) in later milestones.
add_library(rend_renderer INTERFACE)
add_library(rend::renderer ALIAS rend_renderer)
target_include_directories(rend_renderer INTERFACE include)
target_link_libraries(rend_renderer INTERFACE rend::platform rend::gpu)

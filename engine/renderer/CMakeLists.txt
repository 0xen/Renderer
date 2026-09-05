# The renderer layer's first real piece: the command/event message queue
# between producers (app, later a Python host) and the frame loop
# (docs/ARCHITECTURE.md "Renderer layer"). Frame graph, residency and the
# descriptor model land here in later milestones.
add_library(rend_renderer OBJECT
    src/message_queue.cpp
)
add_library(rend::renderer ALIAS rend_renderer)

target_include_directories(rend_renderer PUBLIC include)
target_link_libraries(rend_renderer PUBLIC rend::platform rend::gpu)
target_link_libraries(rend_renderer PRIVATE rend_warnings)
set_target_properties(rend_renderer PROPERTIES FOLDER "engine")

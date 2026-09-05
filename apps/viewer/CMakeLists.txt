add_executable(viewer main.cpp ui.cpp)
target_link_libraries(viewer PRIVATE rend::engine rend::assetio rend_warnings imgui)
add_dependencies(viewer shaders)
set_target_properties(viewer PROPERTIES FOLDER "apps")

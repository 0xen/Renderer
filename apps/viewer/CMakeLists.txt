add_executable(viewer main.cpp)
target_link_libraries(viewer PRIVATE rend::engine rend_warnings)
add_dependencies(viewer shaders)
set_target_properties(viewer PROPERTIES FOLDER "apps")

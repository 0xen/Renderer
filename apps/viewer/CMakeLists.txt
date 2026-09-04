add_executable(viewer main.cpp)
target_link_libraries(viewer PRIVATE rend::engine rend_warnings)
set_target_properties(viewer PROPERTIES FOLDER "apps")

add_executable(viewer main.cpp ui.cpp)
target_link_libraries(viewer PRIVATE rend::engine rend::assetio rend_warnings imgui meshoptimizer)
# Header only (C-ABI typedefs): the Python host DLL is loaded at runtime
# when a scene declares scripts, never linked — scenes without scripts
# must run on installs without any Python bits.
target_include_directories(viewer PRIVATE ${PROJECT_SOURCE_DIR}/pyhost/include)
add_dependencies(viewer shaders)
set_target_properties(viewer PROPERTIES FOLDER "apps")

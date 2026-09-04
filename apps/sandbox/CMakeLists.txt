add_executable(sandbox main.cpp)
target_link_libraries(sandbox PRIVATE rend::platform rend_warnings)
set_target_properties(sandbox PROPERTIES FOLDER "apps")

add_executable(interior_gen main.cpp)
target_link_libraries(interior_gen PRIVATE rend_warnings)
set_target_properties(interior_gen PROPERTIES FOLDER "apps")

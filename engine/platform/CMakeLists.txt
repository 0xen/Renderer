add_library(rend_platform OBJECT
    src/backend_factory.cpp
    backends/sdl3/sdl3_backend.cpp
)
add_library(rend::platform ALIAS rend_platform)

target_include_directories(rend_platform PUBLIC include)
target_link_libraries(rend_platform
    PUBLIC rend::core
    PRIVATE rend_warnings SDL3::SDL3-static
)
set_target_properties(rend_platform PROPERTIES FOLDER "engine")

add_library(rend_core STATIC
    src/log.cpp
)
add_library(rend::core ALIAS rend_core)

target_include_directories(rend_core PUBLIC include)
target_link_libraries(rend_core PRIVATE rend_warnings)
set_target_properties(rend_core PROPERTIES FOLDER "engine")

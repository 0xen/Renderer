# Object library: compiled into the single rend engine DLL (engine/CMakeLists.txt).
add_library(rend_core OBJECT
    src/log.cpp
    src/paths.cpp
)
add_library(rend::core ALIAS rend_core)

target_include_directories(rend_core PUBLIC include)
target_link_libraries(rend_core PRIVATE rend_warnings)
set_target_properties(rend_core PROPERTIES FOLDER "engine")

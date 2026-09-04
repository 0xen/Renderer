# The engine ships as one DLL (rend.dll). The per-layer object libraries
# keep the strict internal dependency structure; their objects are linked
# into this single shared target. All global symbols are exported.
add_subdirectory(core)
add_subdirectory(platform)
add_subdirectory(gpu)
add_subdirectory(renderer)

add_library(rend_engine SHARED)
add_library(rend::engine ALIAS rend_engine)
target_link_libraries(rend_engine
    PUBLIC rend::core rend::platform rend::gpu rend::renderer
)
set_target_properties(rend_engine PROPERTIES
    OUTPUT_NAME "rend"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    FOLDER "engine"
)

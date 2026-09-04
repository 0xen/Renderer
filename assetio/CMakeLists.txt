# Asset IO — a project deliberately separate from the engine. It reads asset
# files (scene XML, model formats) into plain CPU-side data structs; the
# engine is *fed* that data and never touches file formats. Ships as its own
# DLL alongside rend.dll. New formats plug in as importers (see importer.h);
# nothing outside this project changes.
add_library(rend_assetio SHARED
    src/gltf_importer.cpp
    src/importer_registry.cpp
    src/scene_loader.cpp
    src/texture_loader.cpp
)
add_library(rend::assetio ALIAS rend_assetio)

target_include_directories(rend_assetio PUBLIC include)
target_link_libraries(rend_assetio
    # Engine dependency is core utilities only (log, Result); the engine
    # never links back.
    PUBLIC rend::engine
    PRIVATE rend_warnings pugixml::pugixml cgltf stb
)
set_target_properties(rend_assetio PROPERTIES
    OUTPUT_NAME "rend_assetio"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    FOLDER "assetio"
)

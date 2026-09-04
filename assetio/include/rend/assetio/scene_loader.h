#pragma once

#include "rend/assetio/importer.h"
#include "rend/assetio/scene_data.h"
#include "rend/core/result.h"

#include <filesystem>

namespace rend::assetio {

// Parses a scene description XML (docs/SCENE_FORMAT.md). Mesh paths are
// resolved against the XML's directory; pipeline paths are left as written
// (they resolve against the engine's data root, not the scene).
Result<SceneDesc> parseScene(const std::filesystem::path& xmlFile);

// parseScene + imports every referenced model through the registry.
Result<LoadedScene> loadScene(const std::filesystem::path& xmlFile,
                              const ImporterRegistry& importers);

} // namespace rend::assetio

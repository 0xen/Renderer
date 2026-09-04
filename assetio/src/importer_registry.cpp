#include "rend/assetio/importer.h"

#include "gltf_importer.h"

#include <format>

namespace rend::assetio {

ImporterRegistry ImporterRegistry::withBuiltins() {
    ImporterRegistry registry;
    registry.add(std::make_unique<GltfImporter>());
    return registry;
}

void ImporterRegistry::add(std::unique_ptr<IModelImporter> importer) {
    importers_.push_back(std::move(importer));
}

Result<ModelData> ImporterRegistry::import(const std::filesystem::path& file) const {
    for (const auto& importer : importers_) {
        if (importer->supports(file)) {
            return importer->import(file);
        }
    }
    return Error{std::format("No importer for '{}'", file.string())};
}

} // namespace rend::assetio

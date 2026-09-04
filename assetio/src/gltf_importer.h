#pragma once

#include "rend/assetio/importer.h"

namespace rend::assetio {

// glTF 2.0 via cgltf (.gltf / .glb). Node hierarchies are flattened: every
// node's world transform is baked into its mesh's vertex data.
class GltfImporter final : public IModelImporter {
public:
    const char* formatName() const override { return "glTF 2.0"; }
    bool supports(const std::filesystem::path& file) const override;
    Result<ModelData> import(const std::filesystem::path& file) const override;
};

} // namespace rend::assetio

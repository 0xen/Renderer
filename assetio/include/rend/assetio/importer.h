#pragma once

#include "rend/assetio/scene_data.h"
#include "rend/core/result.h"

#include <filesystem>
#include <memory>
#include <vector>

namespace rend::assetio {

// One model file format. Adding a format means implementing this interface
// and registering it — nothing else in the system changes.
class IModelImporter {
public:
    virtual ~IModelImporter() = default;
    virtual const char* formatName() const = 0;
    virtual bool supports(const std::filesystem::path& file) const = 0;
    virtual Result<ModelData> import(const std::filesystem::path& file) const = 0;
};

// Dispatches a file to the first importer that supports it.
class ImporterRegistry {
public:
    // All built-in importers registered (currently: glTF).
    static ImporterRegistry withBuiltins();

    void add(std::unique_ptr<IModelImporter> importer);
    Result<ModelData> import(const std::filesystem::path& file) const;

private:
    std::vector<std::unique_ptr<IModelImporter>> importers_;
};

} // namespace rend::assetio

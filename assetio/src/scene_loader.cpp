#include "rend/assetio/scene_loader.h"

#include "rend/core/log.h"

#include <pugixml.hpp>

#include <format>
#include <sstream>

namespace rend::assetio {

namespace {

// Attributes like position="1 2 3": whitespace-separated floats. Missing
// attributes keep the caller's default; malformed ones are load errors.
template <std::size_t N>
Result<void> parseFloats(const pugi::xml_attribute& attr, std::array<float, N>& out,
                         const char* what) {
    if (!attr) {
        return {};
    }
    std::istringstream stream(attr.value());
    std::array<float, N> values{};
    for (std::size_t i = 0; i < N; ++i) {
        if (!(stream >> values[i])) {
            return Error{std::format("Scene XML: '{}' expects {} floats, got '{}'", what, N,
                                     attr.value())};
        }
    }
    out = values;
    return {};
}

Result<void> parseTransform(const pugi::xml_node& node, TransformDesc& out) {
    if (!node) {
        return {};
    }
    if (auto r = parseFloats(node.attribute("position"), out.position, "Transform position"); !r) {
        return r;
    }
    if (auto r = parseFloats(node.attribute("rotation"), out.rotationDegrees, "Transform rotation");
        !r) {
        return r;
    }
    if (auto r = parseFloats(node.attribute("scale"), out.scale, "Transform scale"); !r) {
        return r;
    }
    return {};
}

} // namespace

Result<SceneDesc> parseScene(const std::filesystem::path& xmlFile) {
    pugi::xml_document doc;
    const pugi::xml_parse_result parsed = doc.load_file(xmlFile.c_str());
    if (!parsed) {
        return Error{std::format("Scene XML parse failed for '{}': {} (offset {})",
                                 xmlFile.string(), parsed.description(), parsed.offset)};
    }

    const pugi::xml_node root = doc.child("Scene");
    if (!root) {
        return Error{std::format("'{}' has no <Scene> root element", xmlFile.string())};
    }

    SceneDesc scene;
    scene.name = root.attribute("name").as_string(xmlFile.stem().string().c_str());

    if (const pugi::xml_node camera = root.child("Camera")) {
        if (auto r = parseFloats(camera.attribute("position"), scene.camera.position,
                                 "Camera position");
            !r) {
            return r.error();
        }
        if (auto r = parseFloats(camera.attribute("target"), scene.camera.target, "Camera target");
            !r) {
            return r.error();
        }
        scene.camera.fovDegrees = camera.attribute("fovDegrees").as_float(scene.camera.fovDegrees);
    }

    const std::filesystem::path baseDir = xmlFile.parent_path();
    for (const pugi::xml_node model : root.children("Model")) {
        ModelNodeDesc desc;
        desc.name = model.attribute("name").as_string("");
        if (desc.name.empty()) {
            return Error{std::format("'{}': <Model> without a name", xmlFile.string())};
        }
        desc.pipelinePath = model.child("Shader").attribute("path").as_string("");
        const char* meshPath = model.child("Mesh").attribute("path").as_string("");
        if (*meshPath == '\0') {
            return Error{std::format("Model '{}' has no <Mesh path=...>", desc.name)};
        }
        desc.meshPath = std::filesystem::absolute(baseDir / meshPath);
        if (auto r = parseTransform(model.child("Transform"), desc.transform); !r) {
            return r.error();
        }
        scene.models.push_back(std::move(desc));
    }
    if (scene.models.empty()) {
        log::warn("Scene '{}' declares no models", scene.name);
    }
    return scene;
}

Result<LoadedScene> loadScene(const std::filesystem::path& xmlFile,
                              const ImporterRegistry& importers) {
    auto sceneResult = parseScene(xmlFile);
    if (!sceneResult) {
        return sceneResult.error();
    }
    SceneDesc& desc = sceneResult.value();

    LoadedScene loaded;
    loaded.name = std::move(desc.name);
    loaded.camera = desc.camera;
    for (ModelNodeDesc& model : desc.models) {
        auto imported = importers.import(model.meshPath);
        if (!imported) {
            return Error{std::format("Model '{}': {}", model.name, imported.error().message)};
        }
        loaded.models.push_back(
            {.desc = std::move(model), .data = std::move(imported).value()});
    }
    return loaded;
}

} // namespace rend::assetio

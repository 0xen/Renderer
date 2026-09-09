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
    // loading="wait" (default) holds the scene behind a loading screen;
    // "streaming" shows it immediately with assets popping in as they land.
    const std::string loading = root.attribute("loading").as_string("wait");
    if (loading == "streaming") {
        scene.loading = SceneLoadingMode::Streaming;
    } else if (loading != "wait") {
        return Error{std::format("'{}': unknown loading mode '{}' (wait|streaming)",
                                 xmlFile.string(), loading)};
    }

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
        if (const pugi::xml_attribute flyFrom = camera.attribute("flyFrom")) {
            if (auto r = parseFloats(flyFrom, scene.camera.flyFrom, "Camera flyFrom"); !r) {
                return r.error();
            }
            scene.camera.flySeconds = camera.attribute("flySeconds").as_float(8.0f);
        }
    }

    for (const pugi::xml_node light : root.children("Light")) {
        LightDesc desc;
        const char* type = light.attribute("type").as_string("directional");
        if (std::string_view(type) == "directional") {
            desc.type = LightType::Directional;
            if (auto r =
                    parseFloats(light.attribute("direction"), desc.direction, "Light direction");
                !r) {
                return r.error();
            }
            desc.castsShadows = light.attribute("castsShadows").as_bool(true);
        } else if (std::string_view(type) == "point") {
            desc.type = LightType::Point;
            if (auto r =
                    parseFloats(light.attribute("position"), desc.position, "Light position");
                !r) {
                return r.error();
            }
            desc.radius = light.attribute("radius").as_float(desc.radius);
            // Shadows are opt-in for point lights: each shadowing light
            // costs a capture (raster) or rays (traced).
            desc.castsShadows = light.attribute("castsShadows").as_bool(false);
        } else {
            return Error{std::format(
                "'{}': unsupported light type '{}' (directional or point)", xmlFile.string(),
                type)};
        }
        if (auto r = parseFloats(light.attribute("color"), desc.color, "Light color"); !r) {
            return r.error();
        }
        desc.intensity = light.attribute("intensity").as_float(desc.intensity);
        scene.lights.push_back(desc);
    }

    for (const pugi::xml_node probe : root.children("ReflectionProbe")) {
        ReflectionProbeDesc desc;
        if (auto r = parseFloats(probe.attribute("position"), desc.position,
                                 "ReflectionProbe position");
            !r) {
            return r.error();
        }
        scene.reflectionProbes.push_back(desc);
    }

    if (const pugi::xml_node fog = root.child("Fog")) {
        scene.fog.enabled = true;
        if (auto r = parseFloats(fog.attribute("position"), scene.fog.position, "Fog position");
            !r) {
            return r.error();
        }
        if (auto r = parseFloats(fog.attribute("size"), scene.fog.size, "Fog size"); !r) {
            return r.error();
        }
        if (auto r = parseFloats(fog.attribute("color"), scene.fog.color, "Fog color"); !r) {
            return r.error();
        }
        scene.fog.density = fog.attribute("density").as_float(scene.fog.density);
        scene.fog.anisotropy = fog.attribute("anisotropy").as_float(scene.fog.anisotropy);
        scene.fog.steps = fog.attribute("steps").as_float(scene.fog.steps);
    }

    const std::filesystem::path baseDir = xmlFile.parent_path();
    for (const pugi::xml_node model : root.children("Model")) {
        ModelNodeDesc desc;
        desc.name = model.attribute("name").as_string("");
        if (desc.name.empty()) {
            return Error{std::format("'{}': <Model> without a name", xmlFile.string())};
        }
        desc.pipelinePath = model.child("Shader").attribute("path").as_string("");
        desc.reflective = model.attribute("reflective").as_bool(false);
        desc.lodEnabled = model.attribute("lod").as_bool(true);
        // Optional scene-local fragment shader override: precompiled
        // SPIR-V shipped WITH the scene (offline dxc, never runtime),
        // resolved against the XML's folder like mesh paths.
        if (const char* fragment = model.child("Shader").attribute("fragment").as_string("");
            *fragment != '\0') {
            desc.fragmentShaderPath = std::filesystem::absolute(baseDir / fragment);
        }
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
    // Optional scene scripts, resolved like mesh paths. Parsing only —
    // whether/how they run is the app's business.
    for (const pugi::xml_node script : root.children("Script")) {
        const char* path = script.attribute("path").as_string("");
        if (*path == '\0') {
            return Error{std::format("'{}': <Script> without a path", xmlFile.string())};
        }
        scene.scripts.push_back(std::filesystem::absolute(baseDir / path));
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
    loaded.loading = desc.loading;
    loaded.camera = desc.camera;
    loaded.lights = std::move(desc.lights);
    loaded.reflectionProbes = std::move(desc.reflectionProbes);
    loaded.fog = desc.fog;
    loaded.scripts = std::move(desc.scripts);
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

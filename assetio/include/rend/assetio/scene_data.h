#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// Plain CPU-side data produced by the asset-IO layer. No GPU or renderer
// types appear here: the engine is fed these structs and owns everything
// downstream (upload, residency, draw submission).
namespace rend::assetio {

struct CameraDesc {
    std::array<float, 3> position{0.0f, 0.0f, 0.0f};
    std::array<float, 3> target{0.0f, 0.0f, -1.0f};
    float fovDegrees = 60.0f;
};

// Lights describe WHAT the light is, never the technique (shadow maps vs
// ray tracing is the renderer's offer — docs/ARCHITECTURE.md). Directional
// only for now; position/range arrive with point/spot types.
struct LightDesc {
    std::array<float, 3> direction{0.3f, -1.0f, 0.2f}; // world space, toward the scene
    std::array<float, 3> color{1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    bool castsShadows = true;
};

struct TransformDesc {
    std::array<float, 3> position{0.0f, 0.0f, 0.0f};
    std::array<float, 3> rotationDegrees{0.0f, 0.0f, 0.0f};
    std::array<float, 3> scale{1.0f, 1.0f, 1.0f};
};

// Textures are carried as file paths (resolved against the model file);
// the renderer's data providers decide when bytes actually load.
struct MaterialData {
    std::string name;
    std::array<float, 4> baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    std::filesystem::path baseColorTexture;         // empty = none
    std::filesystem::path normalTexture;            // empty = none
    std::filesystem::path metallicRoughnessTexture; // empty = none
    bool alphaMasked = false;
    float alphaCutoff = 0.5f;
};

// One drawable chunk: a single material over one vertex/index range.
// Vertex streams are de-interleaved; positions/normals are model-space with
// the source node hierarchy already baked in.
struct MeshData {
    std::string name;
    std::uint32_t materialIndex = 0; // into ModelData::materials
    std::vector<float> positions;    // xyz per vertex
    std::vector<float> normals;      // xyz per vertex; empty if the source has none
    std::vector<float> uvs;          // xy per vertex; empty if the source has none
    std::vector<std::uint32_t> indices;

    std::size_t vertexCount() const { return positions.size() / 3; }
    std::size_t triangleCount() const { return indices.size() / 3; }
};

struct ModelData {
    std::string name;
    std::vector<MaterialData> materials;
    std::vector<MeshData> meshes;
};

// --- Scene description (parsed straight from the scene XML) ---

struct ModelNodeDesc {
    std::string name;
    std::filesystem::path pipelinePath; // engine-data-relative (see SCENE_FORMAT.md)
    std::filesystem::path meshPath;     // resolved to an absolute path
    TransformDesc transform;
};

struct SceneDesc {
    std::string name;
    CameraDesc camera;
    std::vector<LightDesc> lights;
    std::vector<ModelNodeDesc> models;
};

// --- Fully loaded scene: the description plus every imported payload ---

struct LoadedModel {
    ModelNodeDesc desc;
    ModelData data;
};

struct LoadedScene {
    std::string name;
    CameraDesc camera;
    std::vector<LightDesc> lights;
    std::vector<LoadedModel> models;
};

} // namespace rend::assetio

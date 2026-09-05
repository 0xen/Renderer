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

// Where a reflection probe captures its surroundings from. Like lights it
// describes the scene, never the technique: the renderer decides whether a
// cubemap is captured there or reflections are traced instead.
struct ReflectionProbeDesc {
    std::array<float, 3> position{0.0f, 0.0f, 0.0f};
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
    // Alpha-blend material (glTF alphaMode BLEND): drawn see-through;
    // opacity = baseColorFactor.a x base-color texture alpha.
    bool transparent = false;
};

// --- Animation data (node hierarchy, skins, morph targets, keyframes) ---

// One node of an animated model's hierarchy, rest pose as local TRS.
// Parents always precede children, so world matrices resolve in one pass.
struct SkeletonNode {
    std::string name;
    std::int32_t parent = -1; // index into SkeletonData::nodes; -1 = root
    std::array<float, 3> translation{0.0f, 0.0f, 0.0f};
    std::array<float, 4> rotation{0.0f, 0.0f, 0.0f, 1.0f}; // quaternion xyzw
    std::array<float, 3> scale{1.0f, 1.0f, 1.0f};
};

// The full node hierarchy of an animated model plus its skin: jointNodes
// maps skin-local joint indices (what vertices reference) to nodes, with
// one column-major inverse bind matrix each.
struct SkeletonData {
    std::vector<SkeletonNode> nodes;
    std::vector<std::uint32_t> jointNodes;
    std::vector<std::array<float, 16>> inverseBind;

    bool empty() const { return nodes.empty(); }
};

enum class AnimationPath { Translation, Rotation, Scale, Weights };
enum class AnimationInterpolation { Step, Linear, CubicSpline };

// One keyframe stream targeting one node: times paired with 3 (T/S), 4 (R)
// or morph-target-count (Weights) floats per key — three value tuples per
// key (in-tangent, value, out-tangent) when interpolation is CubicSpline.
struct AnimationChannelData {
    std::uint32_t node = 0; // into SkeletonData::nodes
    AnimationPath path = AnimationPath::Translation;
    AnimationInterpolation interpolation = AnimationInterpolation::Linear;
    std::vector<float> times;
    std::vector<float> values;
};

struct AnimationData {
    std::string name;
    float duration = 0.0f; // seconds; max input time over all channels
    std::vector<AnimationChannelData> channels;
};

// Per-morph-target vertex deltas (positions required, normals optional).
struct MorphTargetData {
    std::vector<float> positionDeltas; // xyz per vertex
    std::vector<float> normalDeltas;   // xyz per vertex; empty if none
};

// One drawable chunk: a single material over one vertex/index range.
// Vertex streams are de-interleaved. For rigid models positions/normals
// arrive with the source node hierarchy already baked in; for animated
// models (skeleton non-empty) they stay in mesh space and sourceNode says
// where the mesh hangs in the hierarchy.
struct MeshData {
    std::string name;
    std::uint32_t materialIndex = 0; // into ModelData::materials
    std::vector<float> positions;    // xyz per vertex
    std::vector<float> normals;      // xyz per vertex; empty if the source has none
    std::vector<float> uvs;          // xy per vertex; empty if the source has none
    std::vector<std::uint32_t> indices;

    // Animated models only:
    std::uint32_t sourceNode = 0; // into SkeletonData::nodes
    bool skinned = false;
    std::vector<std::uint16_t> joints; // 4 per vertex, skin-local joint indices
    std::vector<float> weights;        // 4 per vertex, sum ~1
    std::vector<MorphTargetData> morphTargets;
    std::vector<float> morphWeights; // rest weights, one per target

    std::size_t vertexCount() const { return positions.size() / 3; }
    std::size_t triangleCount() const { return indices.size() / 3; }
};

struct ModelData {
    std::string name;
    std::vector<MaterialData> materials;
    std::vector<MeshData> meshes;
    // Non-empty skeleton = animated model: meshes are unbaked (see above).
    SkeletonData skeleton;
    std::vector<AnimationData> animations;
};

// --- Scene description (parsed straight from the scene XML) ---

struct ModelNodeDesc {
    std::string name;
    std::filesystem::path pipelinePath; // engine-data-relative (see SCENE_FORMAT.md)
    // Optional scene-shipped fragment shader (precompiled SPIR-V, resolved
    // absolute): replaces the engine's scene fragment stage — scene-local
    // looks (e.g. toon) without touching the main project.
    std::filesystem::path fragmentShaderPath;
    std::filesystem::path meshPath; // resolved to an absolute path
    TransformDesc transform;
    // Semantic surface tag (intent model: says what the object IS, never
    // the technique): reflective objects may get traced reflections on
    // hardware that offers them; elsewhere they shade as plain surfaces.
    bool reflective = false;
};

struct SceneDesc {
    std::string name;
    CameraDesc camera;
    std::vector<LightDesc> lights;
    // Optional; a probe-capable renderer with none listed picks its own
    // default position (see SCENE_FORMAT.md).
    std::vector<ReflectionProbeDesc> reflectionProbes;
    std::vector<ModelNodeDesc> models;
    // Optional Python scripts to run alongside the scene (<Script path>,
    // resolved to absolute like mesh paths). A scene listing none runs
    // with no Python involvement at all; the app decides what "run"
    // means (assetio only reports the paths — it never executes code).
    std::vector<std::filesystem::path> scripts;
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
    std::vector<ReflectionProbeDesc> reflectionProbes;
    std::vector<LoadedModel> models;
    std::vector<std::filesystem::path> scripts;
};

} // namespace rend::assetio

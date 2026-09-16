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
    // Optional fly-in (<Camera flyFrom="x y z" flySeconds="s">): the
    // camera spawns at flyFrom and eases into the authored position over
    // flySeconds, looking at the target throughout; free flight takes
    // over when it lands. flySeconds <= 0 = no fly-in.
    std::array<float, 3> flyFrom{0.0f, 0.0f, 0.0f};
    float flySeconds = 0.0f;
};

// Lights describe WHAT the light is, never the technique (shadow maps vs
// ray tracing is the renderer's offer — docs/ARCHITECTURE.md).
// Directional: direction/color/intensity/castsShadows. Point: position/
// radius/color/intensity/castsShadows (radius = falloff reach; shadows
// default OFF for point lights — they cost a per-light capture or rays).
enum class LightType { Directional, Point };

struct LightDesc {
    LightType type = LightType::Directional;
    std::array<float, 3> direction{0.3f, -1.0f, 0.2f}; // world space, toward the scene
    std::array<float, 3> position{0.0f, 0.0f, 0.0f};   // point lights only
    float radius = 10.0f;                              // point falloff reach
    std::array<float, 3> color{1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    bool castsShadows = true; // parse default: true directional, false point
};

// Where a reflection probe captures its surroundings from. Like lights it
// describes the scene, never the technique: the renderer decides whether a
// cubemap is captured there or reflections are traced instead.
struct ReflectionProbeDesc {
    std::array<float, 3> position{0.0f, 0.0f, 0.0f};
};

// A box of participating media (<Fog>): position is the box center, size
// its full extents. Like lights it describes the scene, never the
// technique — how the renderer integrates the media (per-pixel march,
// froxels, ...) is its own offer. enabled is false when the scene
// declares no <Fog> element.
struct FogDesc {
    bool enabled = false;
    std::array<float, 3> position{0.0f, 0.0f, 0.0f};
    std::array<float, 3> size{100.0f, 20.0f, 100.0f};
    std::array<float, 3> color{0.75f, 0.8f, 0.87f}; // scattering albedo
    float density = 0.02f;    // extinction per meter inside the box
    float anisotropy = 0.0f;  // Henyey-Greenstein g, -1..1 (0 = isotropic)
    float steps = 24.0f;      // raymarch sample budget hint
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
    std::filesystem::path meshPath;     // resolved to an absolute path
    TransformDesc transform;
    // Semantic surface tag (intent model: says what the object IS, never
    // the technique): reflective objects may get traced reflections on
    // hardware that offers them; elsewhere they shade as plain surfaces.
    bool reflective = false;
    // GPU-driven level-of-detail (<Model lod="off">): on by default; off
    // locks the model's meshes to full detail (no simplified chains are
    // built, the cull pass never swaps their index ranges).
    bool lodEnabled = true;
    // Which animation clip an animated model starts on
    // (<Animation clip="walk"/>): a ModelData::animations name. Empty =
    // the first clip, the historical behaviour. An unknown name warns and
    // falls back to the first clip.
    std::string animationClip;
};

// Scene-wide asset-loading intent (<Scene loading=...>): how the app
// should present the load, never how it schedules it (loading stays
// nonblocking to the renderer either way).
enum class SceneLoadingMode {
    Wait,      // hold the scene behind a loading screen until assets land
    Streaming, // show the scene immediately; assets pop in as they arrive
};

struct SceneDesc {
    std::string name;
    SceneLoadingMode loading = SceneLoadingMode::Wait;
    CameraDesc camera;
    std::vector<LightDesc> lights;
    // Optional; a probe-capable renderer with none listed picks its own
    // default position (see SCENE_FORMAT.md).
    std::vector<ReflectionProbeDesc> reflectionProbes;
    // Optional volumetric fog volume (<Fog>); enabled=false when absent.
    FogDesc fog;
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
    SceneLoadingMode loading = SceneLoadingMode::Wait;
    CameraDesc camera;
    std::vector<LightDesc> lights;
    std::vector<ReflectionProbeDesc> reflectionProbes;
    FogDesc fog;
    std::vector<LoadedModel> models;
    std::vector<std::filesystem::path> scripts;
};

} // namespace rend::assetio

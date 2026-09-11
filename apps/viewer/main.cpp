#include "rend/assetio/scene_loader.h"
#include "rend/assetio/texture_loader.h"
#include "rend/core/log.h"
#include "rend/core/math.h"
#include "rend/core/paths.h"
#include "rend/core/profile.h"
#include "rend/gpu/acceleration_structure.h"
#include "rend/gpu/descriptor_table.h"
#include "rend/gpu/texture_uploader.h"
#include "rend/gpu/device.h"
#include "rend/gpu/frame_renderer.h"
#include "rend/gpu/instance.h"
#include "rend/gpu/pipeline.h"
#include "rend/gpu/probe_capture.h"
#include "rend/gpu/shader.h"
#include "rend/gpu/swapchain.h"
#include "rend/gpu/memory_pool.h"
#include "rend/gpu/transfer.h"
#include "rend/platform/backend.h"
#include "rend/pyhost/pyhost.h"
#include "rend/renderer/message_queue.h"

#include <meshoptimizer.h>

#include "ui.h"

// LoadLibrary only — the Python host DLL is optional at runtime.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <imgui.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <functional>
#include <format>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>

using namespace rend;

namespace {

// Where a mesh landed in the geometry pool; becomes one indirect entry.
struct GeometryLocation {
    gpu::BufferSlice vertices;
    gpu::BufferSlice indices;
    std::uint32_t indexCount = 0;
    std::uint32_t materialIndex = 0;
};

// Headroom for runtime-spawned objects: buffers and per-slot indirect
// regions are sized for this many extra draw entries beyond the scene's,
// so runtime loads never recreate buffers or rewrite non-bindless
// descriptors. Exceeding it fails the load with an event.
constexpr std::uint32_t kRuntimeObjectCapacity = 256;

// Per-slot regions of the per-object transform buffer (binding 19). Must
// match kTransformCapacity in shading.hlsli / shadow.hlsl.
constexpr std::uint32_t kTransformCapacity = 4096;

// Instance-row table (binding 20): SV_InstanceID resolves through these to
// the object/material row and the transform row, so one indirect entry
// with instanceCount N draws N placements of shared geometry. One global
// region (not per-slot) — rows are only rewritten in ways in-flight frames
// tolerate (appends past their instanceCount; swap-remove leaves a benign
// one-frame duplicate). Must match InstanceRow in shading.hlsli.
constexpr std::uint32_t kInstanceRowCapacity = 4096;
struct InstanceRow {
    std::uint32_t objectIndex = 0;
    std::uint32_t transformIndex = 0;
};

// AABB per draw entry for the cull shader's frustum test (binding 22,
// per-slot regions). bmax[3] picks the mode: 0 = world-space box, tested
// once (scene geometry, world-baked); 1 = the mesh's LOCAL box — the
// shader pushes it through each instance's transform and tests PER
// INSTANCE (runtime models), so the CPU never maintains union bounds.
// bmin[3] is a small-integer bitmask (kBounds*, exact in float): bit 0
// marks entries the cull test must never drop (animated meshes — their
// pose can exceed the bind-pose bounds), bit 1 routes the entry to the
// transparent draw stream. Must match ObjectBounds in cull.hlsl.
constexpr float kBoundsAlwaysVisible = 1.0f;
constexpr float kBoundsTransparent = 2.0f;
// Bit 2 (obb.hlsl): the entry's pooled vertices never change after load,
// so the GPU OBB refinement may fit an oriented box from them. Set for
// world-baked, non-animated scene meshes only.
constexpr float kBoundsObbEligible = 4.0f;
struct ObjectBounds {
    std::array<float, 4> bmin{1e30f, 1e30f, 1e30f, 0.0f};
    std::array<float, 4> bmax{-1e30f, -1e30f, -1e30f, 0.0f};
};
// Refined-OBB table (binding 33, obb.hlsl/cull.hlsl/proxy.hlsl): 16-byte
// header (claim counter) + one 64-byte row per draw entry {center+ready,
// 3x axis+half-extent}. Written only by the GPU; zero-seeded at load.
constexpr std::uint64_t kObbHeaderBytes = 16;
constexpr std::uint64_t kObbRowBytes = 64;
constexpr math::Mat4 kIdentityMat4 = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                      0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

// Discrete GPU LOD (binding 25): one table per draw entry; each level is
// a simplified index range into the SAME vertex block (meshopt only drops
// triangles), built at scene load and static after. lods[0] mirrors the
// template's full-detail ranges; errors are absolute world-space
// simplification error, ascending, and the cull shader picks the coarsest
// level whose error still projects under kLodTargetPixels. lodCount <= 1
// (animated meshes, runtime models, <Model lod="off">, meshes under the
// floor) leaves the template untouched. Must match MeshLodTable in
// cull.hlsl.
constexpr std::uint32_t kMaxMeshLods = 4;
struct MeshLodLevel {
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
    float error = 0.0f;
    std::uint32_t pad = 0;
};
struct MeshLodTable {
    std::uint32_t lodCount = 0;
    std::uint32_t pad0 = 0;
    std::uint32_t pad1 = 0;
    std::uint32_t pad2 = 0;
    std::array<MeshLodLevel, kMaxMeshLods> lods{};
};
static_assert(sizeof(MeshLodTable) == 16 + kMaxMeshLods * 16);
// Meshes under this many indices never grow a chain (a few hundred
// triangles simplify into visible mush for no draw-cost win), and a level
// whose target falls under a quarter of it stops the chain.
constexpr std::size_t kLodMinIndices = 1536;
// Simplification error target in screen pixels: a level is only picked
// once its geometric error would project under this size.
constexpr float kLodTargetPixels = 1.0f;

// Maps assetio's backend-agnostic texture payload onto the uploader:
// decoded RGBA8 takes the mip-generating path (color space = the caller's
// role for the texture), pre-compressed payloads copy their stored mips
// with the FILE's declared color space (authoritative for DDS).
Result<std::unique_ptr<gpu::Image>> uploadTextureData(gpu::TextureUploader& uploader,
                                                      const assetio::TextureData& data,
                                                      bool srgbRole) {
    if (data.encoding == assetio::TextureEncoding::Bc7) {
        std::vector<gpu::TextureUploader::CompressedMip> mips(data.mips.size());
        for (std::size_t i = 0; i < data.mips.size(); ++i) {
            mips[i] = {.width = data.mips[i].width,
                       .height = data.mips[i].height,
                       .byteOffset = data.mips[i].byteOffset,
                       .byteLength = data.mips[i].byteLength};
        }
        return uploader.uploadCompressed(data.srgb ? gpu::TextureUploader::kFormatBc7Srgb
                                                   : gpu::TextureUploader::kFormatBc7Unorm,
                                         mips.data(), static_cast<std::uint32_t>(mips.size()),
                                         data.bytes.data(), data.bytes.size());
    }
    return uploader.upload(data.width, data.height, data.bytes.data(), srgbRole);
}

// One row per object in the bindless table's SSBO, found by the indirect
// entry's firstInstance. Must match ObjectData in scene.hlsl.
constexpr std::uint32_t kObjectAlphaMasked = 1u;
constexpr std::uint32_t kObjectTransparent = 2u;
constexpr std::uint32_t kObjectReflective = 4u; // per-object RT reflections
struct ObjectData {
    std::uint32_t textureIndex = 0;
    std::uint32_t normalIndex = 0; // 0 = no normal map (use vertex normal)
    std::uint32_t mrIndex = 0;     // 0 = factors only (glTF: B=metal, G=rough)
    std::uint32_t flags = 0;       // kObject* bits above
    float alphaCutoff = 0.5f;
    float baseAlpha = 1.0f; // baseColorFactor.a: blend opacity multiplier
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
};

// Per-slot camera region. viewProj feeds the raster vertex shader; the
// extra vectors are ray-generation axes for the traced primary pass
// (right/up premultiplied by tan(fov/2)*aspect and tan(fov/2)).
// Must match CameraData in scene.hlsl / rt_primary.hlsl.
struct CameraData {
    math::Mat4 viewProj{};
    std::array<float, 4> position{};
    std::array<float, 4> rightAxis{};
    std::array<float, 4> upAxis{};
    std::array<float, 4> forwardAxis{};
};
static_assert(sizeof(CameraData) == 128);

// Per BLAS-geometry (== object) index: {firstIndex, vertexOffset} in
// uint32/vertex-stride units — the traced pass pulls hit triangles itself.
// Animated meshes point vertexOffset at their posed per-slot region base
// and carry the per-slot vertex stride; the shader adds slot * slotStride
// so traced attributes come from the same pose the BLAS was refitted to.
// Must match the uint4 layout in rt_primary.hlsl (binding 12).
struct GeometryInfo {
    std::uint32_t firstIndex = 0;
    std::uint32_t vertexOffset = 0;
    std::uint32_t slotStride = 0; // vertices per slot copy; 0 = static
    std::uint32_t pad = 0;
};

// Interleaved vertex layout of the scene pass: position, normal, uv.
constexpr std::uint32_t kVertexStride = 8 * sizeof(float);

// Traced passes and runtime models: the TLAS is dynamic (fixed capacity,
// rebuilt in place from a per-slot instance region every frame), so
// runtime model instances and their transforms reach every traced path
// with zero descriptor/recording churn. The world-baked scene BLAS rides
// instance 0 with a sentinel customIndex (geometry index == object index
// there); runtime instances carry a base into the remap table (binding
// 32) because their object rows recycle scattered slots. Sentinel must
// match kRtSceneInstance in rt_common.hlsli.
constexpr std::uint32_t kRtSceneCustomIndex = 0xFFFFFFu;
constexpr std::uint32_t kRtInstanceCapacity = 1 + 256; // scene + runtime instances

constexpr std::uint32_t kShadowMapSize = 2048;
constexpr std::uint32_t kShadowCascades = 4;
constexpr float kPi = 3.14159265358979323846f;

// Reflection probe capture: face resolution and format of the load-time
// cubemap, plus the six extra camera/light buffer regions the capture pass
// indexes with its push-constant slot (appended after the per-frame ones,
// so scene.hlsl needs no changes to render probe faces).
constexpr std::uint32_t kProbeFaceSize = 256;
constexpr std::uint32_t kProbeFormat = 43; // VK_FORMAT_R8G8B8A8_SRGB
// Point-light shadow-distance cubes (soft lamp shadows need little
// resolution; 16 lights x 6 x 256^2 R32F = 24 MiB worst case).
constexpr std::uint32_t kPointShadowFaceSize = 256;
constexpr std::uint32_t kProbeFaces = 6;
constexpr std::uint32_t kCameraRegions = gpu::FrameRenderer::kFramesInFlight + kProbeFaces;

// LightData.reflections values; must match shading.hlsli.
constexpr std::uint32_t kReflectionProbe = 0;
constexpr std::uint32_t kReflectionTraced = 1;
constexpr std::uint32_t kReflectionNone = 2;

math::Vec3 vadd(const math::Vec3& a, const math::Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

// Column-major matrix from a scene <Transform> (XYZ euler degrees).
math::Mat4 transformMatrix(const assetio::TransformDesc& t) {
    const float cx = std::cos(t.rotationDegrees[0] * kPi / 180.0f);
    const float sx = std::sin(t.rotationDegrees[0] * kPi / 180.0f);
    const float cy = std::cos(t.rotationDegrees[1] * kPi / 180.0f);
    const float sy = std::sin(t.rotationDegrees[1] * kPi / 180.0f);
    const float cz = std::cos(t.rotationDegrees[2] * kPi / 180.0f);
    const float sz = std::sin(t.rotationDegrees[2] * kPi / 180.0f);
    // R = Rz * Ry * Rx, columns scaled, translation last.
    math::Mat4 m{};
    m[0] = cz * cy * t.scale[0];
    m[1] = sz * cy * t.scale[0];
    m[2] = -sy * t.scale[0];
    m[4] = (cz * sy * sx - sz * cx) * t.scale[1];
    m[5] = (sz * sy * sx + cz * cx) * t.scale[1];
    m[6] = cy * sx * t.scale[1];
    m[8] = (cz * sy * cx + sz * sx) * t.scale[2];
    m[9] = (sz * sy * cx - cz * sx) * t.scale[2];
    m[10] = cy * cx * t.scale[2];
    m[12] = t.position[0];
    m[13] = t.position[1];
    m[14] = t.position[2];
    m[15] = 1.0f;
    return m;
}

// Column-major TRS matrix from decomposed translation/quaternion/scale.
// Bakes a world matrix into a mesh's CPU data (positions, normals, morph
// deltas) — scene <Transform>s at load, and placement of runtime-spawned
// models. Morph deltas are direction-like: rotate/scale, no translation.
void bakeMeshTransform(assetio::MeshData& mesh, const math::Mat4& m) {
    for (std::size_t v = 0; v + 2 < mesh.positions.size(); v += 3) {
        const float x = mesh.positions[v], y = mesh.positions[v + 1], z = mesh.positions[v + 2];
        mesh.positions[v] = m[0] * x + m[4] * y + m[8] * z + m[12];
        mesh.positions[v + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
        mesh.positions[v + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    }
    // Normals are renormalized per vertex; remember each vertex's factor —
    // morph NORMAL deltas add onto the stored normal at pose time
    // (skin.hlsl normalizes the sum), so they must live in the SAME scaled
    // space or a scaled transform under/over-applies them and morphed
    // normals lag the surface (the animation_lab cube's scale-0.4 bug).
    std::vector<float> invNormalLen(mesh.normals.size() / 3, 1.0f);
    for (std::size_t v = 0; v + 2 < mesh.normals.size(); v += 3) {
        const float x = mesh.normals[v], y = mesh.normals[v + 1], z = mesh.normals[v + 2];
        const math::Vec3 tn{m[0] * x + m[4] * y + m[8] * z, m[1] * x + m[5] * y + m[9] * z,
                            m[2] * x + m[6] * y + m[10] * z};
        const float len = std::sqrt(tn.x * tn.x + tn.y * tn.y + tn.z * tn.z);
        const float inv = len > 0.0f ? 1.0f / len : 0.0f;
        invNormalLen[v / 3] = inv;
        mesh.normals[v] = tn.x * inv;
        mesh.normals[v + 1] = tn.y * inv;
        mesh.normals[v + 2] = tn.z * inv;
    }
    for (auto& target : mesh.morphTargets) {
        for (std::size_t v = 0; v + 2 < target.positionDeltas.size(); v += 3) {
            const float x = target.positionDeltas[v], y = target.positionDeltas[v + 1],
                        z = target.positionDeltas[v + 2];
            target.positionDeltas[v] = m[0] * x + m[4] * y + m[8] * z;
            target.positionDeltas[v + 1] = m[1] * x + m[5] * y + m[9] * z;
            target.positionDeltas[v + 2] = m[2] * x + m[6] * y + m[10] * z;
        }
        for (std::size_t v = 0; v + 2 < target.normalDeltas.size(); v += 3) {
            const float k = v / 3 < invNormalLen.size() ? invNormalLen[v / 3] : 1.0f;
            const float x = target.normalDeltas[v], y = target.normalDeltas[v + 1],
                        z = target.normalDeltas[v + 2];
            target.normalDeltas[v] = (m[0] * x + m[4] * y + m[8] * z) * k;
            target.normalDeltas[v + 1] = (m[1] * x + m[5] * y + m[9] * z) * k;
            target.normalDeltas[v + 2] = (m[2] * x + m[6] * y + m[10] * z) * k;
        }
    }
}

math::Mat4 composeTrs(const std::array<float, 3>& t, const std::array<float, 4>& r,
                      const std::array<float, 3>& s) {
    const float x = r[0], y = r[1], z = r[2], w = r[3];
    math::Mat4 m{};
    m[0] = (1.0f - 2.0f * (y * y + z * z)) * s[0];
    m[1] = (2.0f * (x * y + z * w)) * s[0];
    m[2] = (2.0f * (x * z - y * w)) * s[0];
    m[4] = (2.0f * (x * y - z * w)) * s[1];
    m[5] = (1.0f - 2.0f * (x * x + z * z)) * s[1];
    m[6] = (2.0f * (y * z + x * w)) * s[1];
    m[8] = (2.0f * (x * z + y * w)) * s[2];
    m[9] = (2.0f * (y * z - x * w)) * s[2];
    m[10] = (1.0f - 2.0f * (x * x + y * y)) * s[2];
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
    m[15] = 1.0f;
    return m;
}

math::Mat4 trsMatrix(const assetio::SkeletonNode& node) {
    return composeTrs(node.translation, node.rotation, node.scale);
}

// One animated mesh's GPU wiring: where its bind-pose source, posed
// destination and skin/morph inputs live (offsets in element units).
struct AnimatedMeshEntry {
    std::uint32_t objectIndex = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t srcVertex = 0;
    std::uint32_t dstVertexBase = 0; // slot advances by vertexCount
    std::uint32_t skinVertexOffset = ~0u;
    std::uint32_t morphBase = 0;
    std::uint32_t morphTargetCount = 0;
    std::uint32_t morphWeightOffset = 0;
    std::uint32_t sourceNode = 0;
    std::size_t modelIndex = 0;
};

// CPU-side playback state for one animated model, sampled every frame.
struct AnimatedModelState {
    const assetio::ModelData* data = nullptr;
    std::size_t modelIndex = 0;
    math::Mat4 modelMatrix{};
    std::uint32_t jointBase = 0; // into the shared joint matrix array
    float time = 0.0f;
    std::vector<std::array<float, 3>> t;
    std::vector<std::array<float, 4>> r;
    std::vector<std::array<float, 3>> s;
    std::vector<math::Mat4> world;
};

// Evaluate one channel at `time` into out[components]. Rotation lerps with
// hemisphere correction; CubicSpline falls back to its value tuples.
void sampleChannel(const assetio::AnimationChannelData& channel, float time, float* out,
                   std::size_t components) {
    const bool cubic = channel.interpolation == assetio::AnimationInterpolation::CubicSpline;
    const std::size_t stride = components * (cubic ? 3 : 1);
    const std::size_t valueOffset = cubic ? components : 0;
    const auto& times = channel.times;
    auto keyValues = [&](std::size_t key) {
        return channel.values.data() + key * stride + valueOffset;
    };
    if (times.empty()) {
        return;
    }
    if (time <= times.front() || times.size() == 1) {
        std::copy_n(keyValues(0), components, out);
        return;
    }
    if (time >= times.back()) {
        std::copy_n(keyValues(times.size() - 1), components, out);
        return;
    }
    const auto next = std::upper_bound(times.begin(), times.end(), time);
    const std::size_t k = static_cast<std::size_t>(next - times.begin()) - 1;
    if (channel.interpolation == assetio::AnimationInterpolation::Step) {
        std::copy_n(keyValues(k), components, out);
        return;
    }
    const float u = (time - times[k]) / (times[k + 1] - times[k]);
    const float* a = keyValues(k);
    const float* b = keyValues(k + 1);
    float sign = 1.0f;
    if (channel.path == assetio::AnimationPath::Rotation) {
        const float dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        sign = dot < 0.0f ? -1.0f : 1.0f; // shortest arc
    }
    for (std::size_t c = 0; c < components; ++c) {
        out[c] = a[c] * (1.0f - u) + b[c] * sign * u;
    }
    if (channel.path == assetio::AnimationPath::Rotation) {
        const float len = std::sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2] +
                                    out[3] * out[3]);
        if (len > 0.0f) {
            for (int c = 0; c < 4; ++c) {
                out[c] /= len;
            }
        }
    }
}
math::Vec3 vmul(const math::Vec3& v, float s) { return {v.x * s, v.y * s, v.z * s}; }

// View matrix for one cube face from an explicit screen basis (right, up,
// forward). Not lookAt: the Vulkan cube-face texel layout demands a LEFT-
// handed basis per face (one axis mirrored vs. a normal camera), which no
// up vector can produce — harmless to rasterize since culling is off, and
// exactly what makes sampled directions land on the captured texels.
math::Mat4 faceView(const math::Vec3& eye, const math::Vec3& right, const math::Vec3& up,
                    const math::Vec3& forward) {
    math::Mat4 m{};
    m[0] = right.x;
    m[4] = right.y;
    m[8] = right.z;
    m[12] = -math::dot(right, eye);
    m[1] = up.x;
    m[5] = up.y;
    m[9] = up.z;
    m[13] = -math::dot(up, eye);
    m[2] = -forward.x;
    m[6] = -forward.y;
    m[10] = -forward.z;
    m[14] = math::dot(forward, eye);
    m[15] = 1.0f;
    return m;
}

// Screen basis per cube face, derived from the Vulkan spec's cube-face
// texel mapping: right = the +u axis, up = the -v axis, forward = the face.
struct ProbeFaceBasis {
    math::Vec3 right, up, forward;
};
constexpr ProbeFaceBasis kProbeFaceBases[kProbeFaces] = {
    {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}},  // +X
    {{0, 0, 1}, {0, 1, 0}, {-1, 0, 0}},  // -X
    {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}},  // +Y
    {{1, 0, 0}, {0, 0, 1}, {0, -1, 0}},  // -Y
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},   // +Z
    {{-1, 0, 0}, {0, 1, 0}, {0, 0, -1}}, // -Z
};

// One dynamic point light inside LightData — must match shading.hlsli.
// Rides the per-slot host-visible light buffer, so per-frame updates
// (the script-driven day/night cycle) are a memcpy, never a stall.
inline constexpr std::uint32_t kMaxPointLights = 16;
struct PointLight {
    std::array<float, 4> positionRadius{}; // xyz world position, w falloff radius
    std::array<float, 4> colorIntensity{}; // rgb color, w intensity (0 = off)
    // x = castsShadows (traced rays), y = baked shadow cube written at
    // this slot (raster sampling), zw spare.
    std::array<float, 4> params{};
};

// One region per frame slot in the light buffer (bindless binding 7).
// Must match LightData in scene.hlsl / shadow.hlsl.
struct LightData {
    std::array<math::Mat4, 4> cascadeViewProj{};
    std::array<float, 4> splitDepths{};
    std::array<float, 3> direction{0.0f, -1.0f, 0.0f};
    float intensity = 1.0f;
    std::array<float, 3> color{1.0f, 1.0f, 1.0f};
    float pcfRadius = 1.0f;
    float biasBase = 0.0015f;
    float mapSize = static_cast<float>(kShadowMapSize);
    std::uint32_t cascadeCount = 0;
    std::uint32_t debugTint = 0;
    std::uint32_t rtShadows = 0;
    std::uint32_t reflections = kReflectionProbe; // kReflection* above
    std::uint32_t pad3 = 0;
    std::uint32_t pad4 = 0;
    // Volumetric fog box; fogColor[3] = step count doubles as the enable
    // flag, so the zero-initialized probe-face regions render fog-free.
    std::array<float, 4> fogBoxMin{}; // xyz min corner, w = density
    std::array<float, 4> fogBoxMax{}; // xyz max corner, w = anisotropy
    std::array<float, 4> fogColor{};  // rgb albedo, w = steps (0 = off)
    // Dynamic sky/ambient + point lights; the defaults reproduce the old
    // hardcoded look (probe-face regions rely on them).
    std::array<float, 4> skyColor{0.02f, 0.02f, 0.04f, 0.0f}; // rgb bg, w = light count
    // ambientColor.w = skybox day phase in [0,1); negative = flat skyColor.
    std::array<float, 4> ambientColor{0.30f, 0.32f, 0.36f, -1.0f};
    std::array<PointLight, kMaxPointLights> pointLights{};
};
static_assert(sizeof(LightData) == 1184);

// Live-tunable sun state behind the ImGui panel; direction is stored as
// angles so the sliders stay intuitive.
struct SunControls {
    float azimuthDeg = 30.0f;   // around +Y, 0 = +X
    float elevationDeg = 70.0f; // above horizon, 90 = straight down
    float intensity = 1.0f;
    std::array<float, 3> color{1.0f, 1.0f, 1.0f};
    float pcfRadius = 1.0f;
    float biasBase = 0.0015f;
    float shadowDistance = 60.0f; // view-space reach of the cascades
    bool debugTint = false;
    bool rtShadows = false; // hybrid ray-traced shadows (RayQuery devices)

    math::Vec3 direction() const {
        const float az = azimuthDeg * kPi / 180.0f;
        const float el = elevationDeg * kPi / 180.0f;
        return {std::cos(el) * std::cos(az), -std::sin(el), std::cos(el) * std::sin(az)};
    }

    // The sun is the scene's FIRST directional light; point lights live
    // in their own slots. No directional light = the default sun.
    static assetio::LightDesc sceneSun(const std::vector<assetio::LightDesc>& lights) {
        for (const assetio::LightDesc& light : lights) {
            if (light.type == assetio::LightType::Directional) {
                return light;
            }
        }
        return {};
    }

    static SunControls fromLight(const assetio::LightDesc& light) {
        SunControls out;
        const math::Vec3 d =
            math::normalize({light.direction[0], light.direction[1], light.direction[2]});
        out.elevationDeg = std::asin(std::clamp(-d.y, -1.0f, 1.0f)) * 180.0f / kPi;
        out.azimuthDeg = std::atan2(d.z, d.x) * 180.0f / kPi;
        out.intensity = light.intensity;
        out.color = light.color;
        return out;
    }
};

struct CascadeFit {
    std::array<math::Mat4, 4> viewProj{};
    std::array<float, 4> splits{};
};

// Cascaded shadow fitting: split the camera frustum up to maxDistance
// (log/linear blend), wrap each slice's bounding sphere in a light-space
// ortho, snap the center to the texel grid (kills edge shimmer while
// moving), and pull the near plane back by the scene radius so casters
// outside the slice (roofs, walls) still land in the map.
CascadeFit fitCascades(const math::Vec3& camPos, const math::Vec3& camForward, float fovDegrees,
                       float aspect, const math::Vec3& lightDir, const math::Vec3& sceneMin,
                       const math::Vec3& sceneMax, float maxDistance) {
    CascadeFit out{};
    constexpr float kNear = 0.1f;
    constexpr float kLogBlend = 0.7f;

    const math::Vec3 sceneExtent = math::sub(sceneMax, sceneMin);
    const float sceneRadius = 0.5f * std::sqrt(math::dot(sceneExtent, sceneExtent));
    const math::Vec3 d = math::normalize(lightDir);
    const math::Vec3 lightUp0 = std::fabs(d.y) > 0.99f ? math::Vec3{1.0f, 0.0f, 0.0f}
                                                       : math::Vec3{0.0f, 1.0f, 0.0f};
    const math::Vec3 lightRight = math::normalize(math::cross(d, lightUp0));
    const math::Vec3 lightUp = math::cross(lightRight, d);

    const math::Vec3 forward = camForward;
    const math::Vec3 right = math::normalize(math::cross(forward, {0.0f, 1.0f, 0.0f}));
    const math::Vec3 up = math::cross(right, forward);
    const float tanHalf = std::tan(fovDegrees * kPi / 180.0f * 0.5f);

    float sliceNear = kNear;
    for (std::uint32_t i = 0; i < kShadowCascades; ++i) {
        const float t = static_cast<float>(i + 1) / kShadowCascades;
        const float linear = kNear + (maxDistance - kNear) * t;
        const float logarithmic = kNear * std::pow(maxDistance / kNear, t);
        const float sliceFar = linear * (1.0f - kLogBlend) + logarithmic * kLogBlend;

        // Bounding sphere of the slice's 8 corners.
        math::Vec3 corners[8];
        int corner = 0;
        for (float depth : {sliceNear, sliceFar}) {
            const float hh = tanHalf * depth;
            const float hw = hh * aspect;
            const math::Vec3 at = vadd(camPos, vmul(forward, depth));
            for (int sy = -1; sy <= 1; sy += 2) {
                for (int sx = -1; sx <= 1; sx += 2) {
                    corners[corner++] = vadd(
                        at, vadd(vmul(right, hw * static_cast<float>(sx)),
                                 vmul(up, hh * static_cast<float>(sy))));
                }
            }
        }
        math::Vec3 center{};
        for (const math::Vec3& c : corners) {
            center = vadd(center, vmul(c, 1.0f / 8.0f));
        }
        float radius = 0.0f;
        for (const math::Vec3& c : corners) {
            const math::Vec3 delta = math::sub(c, center);
            radius = std::max(radius, std::sqrt(math::dot(delta, delta)));
        }

        // Texel snap in the light's XY plane.
        const float worldPerTexel = 2.0f * radius / static_cast<float>(kShadowMapSize);
        const float cx =
            std::round(math::dot(center, lightRight) / worldPerTexel) * worldPerTexel;
        const float cy = std::round(math::dot(center, lightUp) / worldPerTexel) * worldPerTexel;
        const float cz = math::dot(center, d);
        const math::Vec3 snapped =
            vadd(vadd(vmul(lightRight, cx), vmul(lightUp, cy)), vmul(d, cz));

        const float backup = sceneRadius;
        const math::Vec3 eye = math::sub(snapped, vmul(d, radius + backup));
        const math::Mat4 view = math::lookAt(eye, snapped, lightUp0);
        const math::Mat4 proj = math::orthographic(-radius, radius, -radius, radius, 0.05f,
                                                   backup + radius * 3.0f);
        out.viewProj[i] = math::mul(proj, view);
        out.splits[i] = sliceFar;
        sliceNear = sliceFar;
    }
    return out;
}

// Free-fly camera: yaw/pitch angles plus position, driven by RMB mouselook
// and WASD/QE. Yaw 0 looks down -Z (matching math::lookAt's convention).
struct FlyCamera {
    math::Vec3 position{};
    float yaw = 0.0f;   // radians, positive turns right (+X)
    float pitch = 0.0f; // radians, positive looks up; clamped near +/-90
    float fovDegrees = 60.0f;

    math::Vec3 forward() const {
        const float cp = std::cos(pitch);
        return {cp * std::sin(yaw), std::sin(pitch), -cp * std::cos(yaw)};
    }

    math::Mat4 viewProj(std::uint32_t width, std::uint32_t height) const {
        const math::Vec3 f = forward();
        const math::Mat4 view = math::lookAt(
            position, {position.x + f.x, position.y + f.y, position.z + f.z}, {0.0f, 1.0f, 0.0f});
        const float aspect =
            height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
        return math::mul(math::perspective(fovDegrees * kPi / 180.0f, aspect, 0.1f, 300.0f), view);
    }

    static FlyCamera fromScene(const assetio::CameraDesc& camera) {
        FlyCamera out;
        out.position = {camera.position[0], camera.position[1], camera.position[2]};
        const math::Vec3 d = math::normalize(math::sub(
            {camera.target[0], camera.target[1], camera.target[2]}, out.position));
        out.yaw = std::atan2(d.x, -d.z);
        out.pitch = std::asin(std::clamp(d.y, -1.0f, 1.0f));
        out.fovDegrees = camera.fovDegrees;
        return out;
    }
};

// Interleave the assetio streams into the vertex layout the scene pass will
// consume: position (3f), normal (3f), uv (2f). Missing streams pad with
// zeros so one pipeline serves every mesh.
std::vector<float> interleave(const assetio::MeshData& mesh) {
    const std::size_t count = mesh.vertexCount();
    std::vector<float> out;
    out.reserve(count * 8);
    for (std::size_t i = 0; i < count; ++i) {
        out.insert(out.end(), {mesh.positions[i * 3], mesh.positions[i * 3 + 1],
                               mesh.positions[i * 3 + 2]});
        if (mesh.normals.size() == count * 3) {
            out.insert(out.end(),
                       {mesh.normals[i * 3], mesh.normals[i * 3 + 1], mesh.normals[i * 3 + 2]});
        } else {
            out.insert(out.end(), {0.0f, 0.0f, 0.0f});
        }
        if (mesh.uvs.size() == count * 2) {
            out.insert(out.end(), {mesh.uvs[i * 2], mesh.uvs[i * 2 + 1]});
        } else {
            out.insert(out.end(), {0.0f, 0.0f});
        }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    bool debug = false;
    bool vsync = true;
    bool staticMode = false;
    bool rtFromStart = false; // start with ray-traced shadows on (if supported)
    bool rtPrimaryFromStart = false; // start with traced primary rays (if supported)
    // Optional cap on the reflection technique (default: best offered).
    bool forceProbeReflections = false;
    // GPU frustum culling in the compaction pass (IndirectCount mode);
    // --nocull turns it off for A/B comparisons.
    bool frustumCull = true;
    // GPU LOD selection in the same pass (scene meshes with baked chains);
    // --nolod locks everything to full detail for A/B comparisons.
    bool lodSelect = true;
    // GPU occlusion culling (proxy-pass visibility, IndirectCount mode);
    // --noocclusion turns it off for A/B comparisons.
    bool occlusionCull = true;
    // Incremental GPU OBB refinement (a few entries per frame refine
    // their AABB into a PCA-fitted oriented box, fully on the GPU);
    // --noobb keeps the plain AABBs for A/B comparisons.
    bool obbRefine = true;
    // Debug overlay rendering the occlusion-proxy AABB boxes as
    // translucent color (Settings checkbox). Toggling swaps the batch's
    // debug pipeline and costs one invalidateStaticRecordings — no
    // per-frame CPU work.
    bool showOcclusionBoxes = false;
    std::uint64_t benchFrames = 0; // non-zero: exit after N frames with a report
    // Streaming test harness: auto-spawn this model at random intervals
    // through the message queue.
    const char* spawnTestPath = nullptr;
    // Caps the draw-submit ladder for testing the fallbacks; the actual mode
    // is still limited by what the device supports.
    auto maxDrawMode = gpu::DrawSubmitMode::IndirectCount;
    const char* scenePath = nullptr;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--debug") {
            debug = true;
        } else if (arg == "--novsync") {
            vsync = false;
        } else if (arg == "--static") {
            staticMode = true;
        } else if (arg == "--rt") {
            rtFromStart = true;
        } else if (arg == "--rtprimary") {
            rtPrimaryFromStart = true;
        } else if (arg == "--reflections" && i + 1 < argc) {
            const std::string_view technique = argv[++i];
            if (technique == "probe") {
                forceProbeReflections = true;
            } else if (technique != "traced") {
                log::error("Unknown --reflections '{}' (probe|traced)", technique);
                return 1;
            }
        } else if (arg == "--nocull") {
            frustumCull = false;
        } else if (arg == "--nolod") {
            lodSelect = false;
        } else if (arg == "--noocclusion") {
            occlusionCull = false;
        } else if (arg == "--noobb") {
            obbRefine = false;
        } else if (arg == "--spawn-test" && i + 1 < argc) {
            spawnTestPath = argv[++i];
        } else if (arg == "--bench" && i + 1 < argc) {
            const std::string_view count = argv[++i];
            std::from_chars(count.data(), count.data() + count.size(), benchFrames);
        } else if (arg == "--draw-mode" && i + 1 < argc) {
            const std::string_view mode = argv[++i];
            if (mode == "count") {
                maxDrawMode = gpu::DrawSubmitMode::IndirectCount;
            } else if (mode == "indirect") {
                maxDrawMode = gpu::DrawSubmitMode::Indirect;
            } else if (mode == "direct") {
                maxDrawMode = gpu::DrawSubmitMode::Direct;
            } else {
                log::error("Unknown --draw-mode '{}' (count|indirect|direct)", mode);
                return 1;
            }
        } else {
            scenePath = arg.data();
        }
    }

    // --debug: mirror the log to a file next to the exe (survives crashes,
    // hangs, and closed consoles) and turn on synchronization validation.
    if (debug) {
        log::mirrorToFile(executableDirectory() / "viewer.log");
    }

    log::info("Renderer viewer v0.1.0{}", debug ? " (debug)" : "");

    // The assetio project turns the scene XML + referenced model files into
    // plain CPU-side data; the viewer feeds it to the GPU below.
    std::optional<assetio::LoadedScene> scene;
    if (scenePath) {
        const auto start = std::chrono::steady_clock::now();
        auto sceneResult = [&] {
            REND_PROFILE_ZONE("LoadScene");
            return assetio::loadScene(scenePath, assetio::ImporterRegistry::withBuiltins());
        }();
        if (!sceneResult) {
            log::error("Scene load failed: {}", sceneResult.error().message);
            return 1;
        }
        scene = std::move(sceneResult).value();
        const auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                  start)
                .count();
        std::size_t vertices = 0, triangles = 0;
        for (const auto& model : scene->models) {
            std::size_t modelVerts = 0, modelTris = 0;
            for (const auto& mesh : model.data.meshes) {
                modelVerts += mesh.vertexCount();
                modelTris += mesh.triangleCount();
            }
            log::info("  Model '{}': {} meshes, {} materials, {} vertices, {} triangles",
                      model.desc.name, model.data.meshes.size(), model.data.materials.size(),
                      modelVerts, modelTris);
            vertices += modelVerts;
            triangles += modelTris;
        }
        log::info("Scene '{}' loaded in {} ms: {} models, {} vertices, {} triangles", scene->name,
                  ms, scene->models.size(), vertices, triangles);

        // Bake each model's scene <Transform>; animated models additionally
        // arrive with a live hierarchy and unbaked meshes, so pose them at
        // rest on the CPU until the GPU skinning pass lands (joints/
        // weights/morphs ride along unused for now).
        for (auto& model : scene->models) {
            const math::Mat4 modelMatrix = transformMatrix(model.desc.transform);
            std::vector<math::Mat4> world;
            if (!model.data.skeleton.empty()) {
                const auto& nodes = model.data.skeleton.nodes;
                world.resize(nodes.size());
                for (std::size_t i = 0; i < nodes.size(); ++i) {
                    const math::Mat4 local = trsMatrix(nodes[i]);
                    world[i] =
                        nodes[i].parent >= 0
                            ? math::mul(world[static_cast<std::size_t>(nodes[i].parent)], local)
                            : local;
                }
                log::info("  '{}' posed at rest: {} nodes, {} joints, {} animations",
                          model.desc.name, nodes.size(), model.data.skeleton.jointNodes.size(),
                          model.data.animations.size());
            }
            for (auto& mesh : model.data.meshes) {
                if (mesh.skinned) {
                    // Skinned vertices stay in bind space: the skinning
                    // pass poses them with model * world * inverseBind.
                    continue;
                }
                const math::Mat4 m = world.empty()
                                         ? modelMatrix
                                         : math::mul(modelMatrix, world[mesh.sourceNode]);
                bakeMeshTransform(mesh, m);
            }
        }
    } else {
        log::info("No scene file given "
                  "(usage: viewer [--debug] [--novsync] [--static] [--bench N] "
                  "[--draw-mode count|indirect|direct] [--spawn-test model.gltf] <scene.xml>)");
    }

    auto backendResult = platform::createBackend(platform::BackendKind::SDL3);
    if (!backendResult) {
        log::error("Failed to create backend: {}", backendResult.error().message);
        return 1;
    }
    auto backend = std::move(backendResult).value();
    if (auto init = backend->initialize(); !init) {
        log::error("Backend init failed: {}", init.error().message);
        return 1;
    }

    auto instanceResult = gpu::Instance::create({
        .appName = "Renderer Viewer",
        .enableSyncValidation = debug,
        .extraExtensions = backend->requiredVulkanInstanceExtensions(),
    });
    if (!instanceResult) {
        log::error("Vulkan instance creation failed: {}", instanceResult.error().message);
        return 1;
    }
    auto instance = std::move(instanceResult).value();

    auto features = gpu::FeatureSet::gpuDriven();
    features.requiredExtensions.push_back(gpu::kSwapchainExtension);
    auto deviceResult = gpu::Device::create(*instance, features);
    if (!deviceResult) {
        log::error("Vulkan device creation failed: {}", deviceResult.error().message);
        return 1;
    }
    auto device = std::move(deviceResult).value();

    // Milestone 7 step 1: everything the scene pass will draw lives in one
    // device-local memory pool, filled through the transfer queue. Indirect
    // draw entries over these slices come next.
    std::unique_ptr<gpu::MemoryPool> geometryPool;
    std::unique_ptr<gpu::Buffer> indirectBuffer;
    std::unique_ptr<gpu::Buffer> compactedBuffer;
    std::unique_ptr<gpu::Buffer> countBuffer;
    std::unique_ptr<gpu::Buffer> cameraBuffer; // one CameraData per frame slot, CPU-written
    std::unique_ptr<gpu::Buffer> geometryInfoBuffer; // per-object triangle lookup (RT primary)
    std::unique_ptr<gpu::Buffer> rtRemapBuffer; // runtime hit->object rows (binding 32)
    // Animation: static inputs (staged) + per-slot CPU-written state.
    std::unique_ptr<gpu::Buffer> skinVertexBuffer;  // packed joints+weights per skinned vertex
    std::unique_ptr<gpu::Buffer> morphDeltaBuffer;  // pos+normal deltas per vertex/target
    std::unique_ptr<gpu::Buffer> jointBuffer;       // joint matrices per frame slot
    std::unique_ptr<gpu::Buffer> morphWeightBuffer; // sampled weights per frame slot
    std::vector<AnimatedMeshEntry> animatedMeshes;
    std::vector<AnimatedModelState> animatedStates;
    std::vector<math::Mat4> jointMatricesCpu;
    std::vector<float> morphWeightsFrame;
    std::uint32_t totalJoints = 0;
    std::unique_ptr<gpu::Buffer> lightBuffer;  // one LightData per frame slot, CPU-written
    std::vector<std::unique_ptr<gpu::Image>> shadowMaps; // one per cascade
    std::unique_ptr<gpu::AccelerationStructure> blas, tlas; // RT shadow BVH
    // Per-slot BLAS refit inputs: the build geometry list with animated
    // entries pointing at that slot's posed vertex region (empty = static).
    std::vector<std::vector<gpu::AccelerationStructure::TriangleGeometry>> refitGeometries;
    math::Vec3 sceneMin{1e30f, 1e30f, 1e30f};
    math::Vec3 sceneMax{-1e30f, -1e30f, -1e30f};
    // Static <Model> bounds alone, snapshotted after scene setup: the
    // base the cascade-fit AABB is recomputed from when runtime
    // instances spawn/move/unload (sceneMin/Max above hold the union).
    math::Vec3 staticSceneMin{1e30f, 1e30f, 1e30f};
    math::Vec3 staticSceneMax{-1e30f, -1e30f, -1e30f};
    std::vector<gpu::DrawIndexedIndirect> draws; // outlives the loop: Direct mode records from it
    std::unique_ptr<gpu::Buffer> objectBuffer;
    std::unique_ptr<gpu::DescriptorTable> descriptorTable;
    std::vector<std::unique_ptr<gpu::Image>> textures;
    // Scene textures registered for the bindless table; slot-indexed,
    // entry 0 is the 1x1 white fallback. Main-scope: the streaming decode
    // workers and the frame loop's upload pump both read it.
    struct TextureRequest {
        std::filesystem::path path;
        bool srgb = true; // normal/MR maps decode linear (UNORM)
    };
    std::vector<TextureRequest> texturePaths{{}};
    // Nonblocking scene-texture streaming (<Scene loading=...>): decode
    // runs on worker threads while the frame loop renders; the loop
    // uploads a few finished textures per frame and batches the bindless
    // slot rewrites under a waitIdle sync point — every slot points at
    // white until its real texture lands, and a slot being rewritten was
    // sampled by in-flight frames, so the rewrite must not race pending
    // work (UPDATE_AFTER_BIND only covers descriptors no pending command
    // buffer consumes).
    struct SceneTextureStream {
        std::mutex mutex;
        std::deque<std::pair<std::size_t, Result<assetio::TextureData>>> ready;
        std::vector<std::thread> workers;
        std::atomic<std::size_t> next{1};
        std::atomic<bool> quit{false};
        std::vector<std::uint32_t> pendingSlots; // uploaded, awaiting slot rewrite
        std::size_t total = 0;                   // beyond the white fallback
        std::size_t uploaded = 0;
        std::uint64_t texelBytes = 0;
        std::chrono::steady_clock::time_point start;
        std::chrono::steady_clock::time_point lastFlush;
        bool active = false;
    };
    SceneTextureStream texStream;
    // loading="wait": hold the scene behind the loading screen until the
    // stream completes (the renderer keeps pumping frames either way).
    bool waitForTextures = false;
    // Reflection probe capture, deferred until the texture stream
    // completes so the capture samples real textures; assigned during
    // scene setup, invoked once by the frame loop's streaming pump.
    std::function<void()> capturePendingProbe;
    std::vector<GeometryLocation> geometry;
    // Kept alive past load for the runtime model-load path (message queue):
    // the transfer context and texture uploader do the GPU work, the
    // texture map dedups repeat spawns, objectData mirrors the GPU SSBO.
    std::unique_ptr<gpu::TransferContext> transfer;
    std::unique_ptr<gpu::TextureUploader> uploader;
    std::unordered_map<std::string, std::uint32_t> textureSlotByPath;
    std::vector<ObjectData> objectData;
    std::uint32_t templateCapacity = 0; // indirect entries per slot region
    // Canonical per-object world matrices, copied into the current slot's
    // transform-buffer region every frame (identity for scene geometry,
    // which is world-baked; runtime models are placed/moved through them).
    std::unique_ptr<gpu::Buffer> transformBuffer;
    std::vector<math::Mat4> objectTransforms;
    // Instance rows (binding 20): canonical CPU copy + host-visible buffer.
    // Scene draws occupy an identity prefix (row i = {i, i}); runtime
    // resources allocate contiguous per-mesh blocks above it.
    std::unique_ptr<gpu::Buffer> instanceRowBuffer;
    std::vector<InstanceRow> instanceRows;
    // Frustum culling: canonical per-draw-entry world AABBs, copied into
    // the current slot's region of the bounds buffer every frame (same
    // pattern as the transforms), and the device-local frustum-culled
    // draw stream the scene pass draws in IndirectCount mode.
    std::unique_ptr<gpu::Buffer> boundsBuffer;
    std::unique_ptr<gpu::Buffer> culledBuffer;
    std::unique_ptr<gpu::Buffer> transparentBuffer;
    std::vector<ObjectBounds> objectBounds;
    // Per-entry LOD chains (binding 25), one table per scene mesh in draw
    // order; runtime rows past the scene's stay zeroed = full detail.
    std::unique_ptr<gpu::Buffer> meshLodBuffer;
    std::vector<MeshLodTable> meshLodTables;
    // Occlusion visibility (binding 26): per-slot regions the proxy pass
    // marks and the next frame's cull dispatch reads. GPU-only after the
    // all-ones seed (frame 0 must draw everything).
    std::unique_ptr<gpu::Buffer> visibilityBuffer;
    // Refined-OBB table (binding 33): GPU-written oriented boxes the cull
    // and proxy passes prefer over the AABBs once each row is ready.
    std::unique_ptr<gpu::Buffer> obbBuffer;
    // Canonical-row -> draw-entry map (binding 34): the per-instance
    // occlusion proxy pass resolves rows to local bounds through it.
    std::unique_ptr<gpu::Buffer> rowEntryBuffer;
    const auto importers = assetio::ImporterRegistry::withBuiltins();
    // Capability offer from the gpu layer: which shadow techniques this
    // device can run. The UI is built from this list, never from
    // hard-coded assumptions.
    const std::vector<gpu::ShadowTechnique> shadowOffers = device->supportedShadowTechniques();
    const bool rtSupported =
        std::find(shadowOffers.begin(), shadowOffers.end(), gpu::ShadowTechnique::RayTraced) !=
        shadowOffers.end();
    bool rtReady = false; // BVH built and wired into the bindless table
    // Reflection offer, same pattern as shadows: the probe tier is the
    // floor that runs everywhere, ray traced rides the same RT features.
    const std::vector<gpu::ReflectionTechnique> reflectionOffers =
        device->supportedReflectionTechniques();
    std::unique_ptr<gpu::Image> probeImage; // load-time capture cubemap
    bool probeReady = false; // captured and wired into the bindless table
    // Dynamic point lights (declared BEFORE capturePendingProbe so the
    // deferred shadow bake can capture them by reference) + their baked
    // shadow-distance cubes, one per shadow-casting slot.
    std::array<PointLight, kMaxPointLights> pointLights{};
    std::vector<std::unique_ptr<gpu::Image>> pointShadowImages;
    if (scene) {
        const auto start = std::chrono::steady_clock::now();
        // With ray tracing, the geometry pool doubles as the BLAS build
        // input, which needs device-address usage.
        // Storage usage always: the skinning pass poses vertices in place;
        // on RT devices the traced primary pass also reads hit triangles
        // straight out of the pool (descriptor binding 11).
        // 512 MiB: sized for multi-million-triangle stress scenes (Bistro's
        // flattened geometry alone is ~172 MiB interleaved + indices).
        auto poolResult = gpu::MemoryPool::create(
            *device, 512ull * 1024 * 1024,
            gpu::kUsageStorage |
                (rtSupported ? (gpu::kUsageShaderDeviceAddress | gpu::kUsageAccelBuildInput)
                             : 0));
        if (!poolResult) {
            log::error("Memory pool creation failed: {}", poolResult.error().message);
            return 1;
        }
        geometryPool = std::move(poolResult).value();

        auto transferResult = gpu::TransferContext::create(*device);
        if (!transferResult) {
            log::error("Transfer context creation failed: {}", transferResult.error().message);
            return 1;
        }
        transfer = std::move(transferResult).value();

        // Slot 0 of the bindless texture array is a 1x1 white fallback so
        // untextured materials sample neutrally; index 0 doubles as "no
        // normal / no metallic-roughness map". (texturePaths lives at main
        // scope — the streaming decode workers read it from the frame
        // loop's lifetime.)
        auto registerTexture = [&](const std::filesystem::path& path, bool srgb) {
            if (path.empty()) {
                return 0u;
            }
            auto [it, inserted] = textureSlotByPath.try_emplace(
                path.string(), static_cast<std::uint32_t>(texturePaths.size()));
            if (inserted) {
                texturePaths.push_back({path, srgb});
            }
            return it->second;
        };

        // Animation wiring accumulated across the upload loop: packed skin
        // vertices, concatenated morph deltas (6 floats per vertex/target)
        // and the rest-pose morph weights.
        struct PackedSkinVertex {
            std::uint32_t joints01 = 0;
            std::uint32_t joints23 = 0;
            std::array<float, 4> weights{};
        };
        std::vector<PackedSkinVertex> skinVertexData;
        std::vector<float> morphDeltaData;
        std::vector<float> morphWeightsCpu;

        std::size_t lodMeshCount = 0;
        std::uint64_t lodExtraIndices = 0;
        for (std::size_t modelIndex = 0; modelIndex < scene->models.size(); ++modelIndex) {
            const auto& model = scene->models[modelIndex];
            for (const auto& mesh : model.data.meshes) {
                ObjectData object;
                if (mesh.materialIndex < model.data.materials.size()) {
                    const auto& material = model.data.materials[mesh.materialIndex];
                    object.flags = (material.alphaMasked ? kObjectAlphaMasked : 0u) |
                                   (material.transparent ? kObjectTransparent : 0u);
                    object.alphaCutoff = material.alphaCutoff;
                    object.baseAlpha = material.baseColorFactor[3];
                    object.metallicFactor = material.metallicFactor;
                    object.roughnessFactor = material.roughnessFactor;
                    object.textureIndex = registerTexture(material.baseColorTexture, true);
                    object.normalIndex = registerTexture(material.normalTexture, false);
                    object.mrIndex = registerTexture(material.metallicRoughnessTexture, false);
                }
                // Semantic scene tag; the RT-variant shaders decide whether
                // to trace it (offer model — non-RT devices just ignore it).
                if (model.desc.reflective) {
                    object.flags |= kObjectReflective;
                }
                objectData.push_back(object);

                // Per-mesh world AABB (positions are world-baked for scene
                // geometry): feeds the cull shader's frustum test and,
                // merged, the scene AABB the cascade fitting uses.
                ObjectBounds meshBounds;
                for (std::size_t v = 0; v + 2 < mesh.positions.size(); v += 3) {
                    meshBounds.bmin[0] = std::min(meshBounds.bmin[0], mesh.positions[v]);
                    meshBounds.bmin[1] = std::min(meshBounds.bmin[1], mesh.positions[v + 1]);
                    meshBounds.bmin[2] = std::min(meshBounds.bmin[2], mesh.positions[v + 2]);
                    meshBounds.bmax[0] = std::max(meshBounds.bmax[0], mesh.positions[v]);
                    meshBounds.bmax[1] = std::max(meshBounds.bmax[1], mesh.positions[v + 1]);
                    meshBounds.bmax[2] = std::max(meshBounds.bmax[2], mesh.positions[v + 2]);
                }
                if (mesh.skinned || !mesh.morphTargets.empty()) {
                    // Animated: the pose may leave the bind-pose bounds —
                    // never frustum-cull these.
                    meshBounds.bmin[3] += kBoundsAlwaysVisible;
                } else {
                    // Static world-baked vertices: the GPU OBB refinement
                    // may fit a tighter oriented box from the pool.
                    meshBounds.bmin[3] += kBoundsObbEligible;
                }
                if ((object.flags & kObjectTransparent) != 0) {
                    // Routed to the blend pass's stream by the cull shader.
                    meshBounds.bmin[3] += kBoundsTransparent;
                }
                objectBounds.push_back(meshBounds);
                sceneMin = {std::min(sceneMin.x, meshBounds.bmin[0]),
                            std::min(sceneMin.y, meshBounds.bmin[1]),
                            std::min(sceneMin.z, meshBounds.bmin[2])};
                sceneMax = {std::max(sceneMax.x, meshBounds.bmax[0]),
                            std::max(sceneMax.y, meshBounds.bmax[1]),
                            std::max(sceneMax.z, meshBounds.bmax[2])};

                const std::vector<float> vertexData = interleave(mesh);
                // Stride alignment keeps vertexOffset (= offset / stride) exact.
                auto vertexSlice =
                    geometryPool->allocate(vertexData.size() * sizeof(float), kVertexStride);
                auto indexSlice =
                    geometryPool->allocate(mesh.indices.size() * sizeof(std::uint32_t), 4);
                if (!vertexSlice || !indexSlice) {
                    log::error("Memory pool allocation failed: {}",
                               (!vertexSlice ? vertexSlice.error() : indexSlice.error()).message);
                    return 1;
                }
                auto stagedVerts =
                    transfer->stage(geometryPool->buffer(), vertexSlice.value().offset,
                                    vertexData.data(), vertexSlice.value().size);
                auto stagedIndices =
                    transfer->stage(geometryPool->buffer(), indexSlice.value().offset,
                                    mesh.indices.data(), indexSlice.value().size);
                if (!stagedVerts || !stagedIndices) {
                    log::error("Staging failed: {}",
                               (!stagedVerts ? stagedVerts.error() : stagedIndices.error()).message);
                    return 1;
                }
                geometry.push_back({.vertices = vertexSlice.value(),
                                    .indices = indexSlice.value(),
                                    .indexCount = static_cast<std::uint32_t>(mesh.indices.size()),
                                    .materialIndex = mesh.materialIndex});

                // Discrete LOD chain: cascade meshopt_simplify over the
                // mesh's own indices (each level starts from the previous
                // one) at 1/4 the index count per step, staging each
                // level's indices as another pool slice over the SAME
                // vertex block. Failures just end the chain — LOD is an
                // optimization, never a load error. Animated meshes are
                // excluded (their templates patch vertexOffset to per-slot
                // posed regions; the pairing must stay untouched).
                MeshLodTable lodTable;
                if (model.desc.lodEnabled && !mesh.skinned && mesh.morphTargets.empty() &&
                    mesh.indices.size() >= kLodMinIndices) {
                    lodTable.lodCount = 1;
                    lodTable.lods[0] = {
                        .firstIndex = static_cast<std::uint32_t>(indexSlice.value().offset /
                                                                 sizeof(std::uint32_t)),
                        .indexCount = static_cast<std::uint32_t>(mesh.indices.size()),
                        .error = 0.0f};
                    const std::size_t vertexCount = mesh.vertexCount();
                    // result_error is relative to the mesh extent; this
                    // scale takes it to absolute (world-baked) units.
                    const float errorScale = meshopt_simplifyScale(mesh.positions.data(),
                                                                   vertexCount, 3 * sizeof(float));
                    std::vector<std::uint32_t> lodIndices = mesh.indices;
                    std::vector<std::uint32_t> simplified;
                    float lastError = 0.0f;
                    for (std::uint32_t level = 1; level < kMaxMeshLods; ++level) {
                        const std::size_t target = mesh.indices.size() >> (2 * level);
                        if (target < kLodMinIndices / 4) {
                            break;
                        }
                        simplified.resize(lodIndices.size());
                        float relError = 0.0f;
                        const std::size_t count = meshopt_simplify(
                            simplified.data(), lodIndices.data(), lodIndices.size(),
                            mesh.positions.data(), vertexCount, 3 * sizeof(float), target,
                            0.25f, 0, &relError);
                        // No meaningful reduction (dense/degenerate
                        // topology): further levels won't do better.
                        if (count == 0 || count >= lodIndices.size() * 9 / 10) {
                            break;
                        }
                        simplified.resize(count);
                        auto lodSlice =
                            geometryPool->allocate(count * sizeof(std::uint32_t), 4);
                        if (!lodSlice) {
                            log::warn("LOD index allocation failed: {}",
                                      lodSlice.error().message);
                            break;
                        }
                        if (auto staged =
                                transfer->stage(geometryPool->buffer(),
                                                lodSlice.value().offset, simplified.data(),
                                                lodSlice.value().size);
                            !staged) {
                            log::warn("LOD index staging failed: {}", staged.error().message);
                            geometryPool->free(lodSlice.value());
                            break;
                        }
                        // Keep errors non-decreasing so the shader's
                        // coarsest-that-fits scan stays well ordered.
                        lastError = std::max(relError * errorScale, lastError);
                        lodTable.lods[lodTable.lodCount++] = {
                            .firstIndex = static_cast<std::uint32_t>(lodSlice.value().offset /
                                                                     sizeof(std::uint32_t)),
                            .indexCount = static_cast<std::uint32_t>(count),
                            .error = lastError};
                        lodExtraIndices += count;
                        lodIndices = std::move(simplified);
                        simplified = {};
                    }
                    if (lodTable.lodCount > 1) {
                        ++lodMeshCount;
                    } else {
                        lodTable = {}; // single level: leave the entry zeroed
                    }
                }
                meshLodTables.push_back(lodTable);

                // Animated meshes additionally get a per-slot destination
                // region the skinning pass poses into; the indirect entry
                // for the slot points there instead of the bind pose.
                if (mesh.skinned || !mesh.morphTargets.empty()) {
                    const auto vertexCount = static_cast<std::uint32_t>(mesh.vertexCount());
                    auto dstSlice = geometryPool->allocate(
                        std::uint64_t{vertexCount} * gpu::FrameRenderer::kFramesInFlight *
                            kVertexStride,
                        kVertexStride);
                    if (!dstSlice) {
                        log::error("Posed-vertex allocation failed: {}", dstSlice.error().message);
                        return 1;
                    }
                    // Seed every slot copy with the bind pose so consumers
                    // (traced attribute fetch, a missing skin pipeline)
                    // never read uninitialized pool memory.
                    for (std::uint32_t slot = 0; slot < gpu::FrameRenderer::kFramesInFlight;
                         ++slot) {
                        if (auto staged = transfer->stage(
                                geometryPool->buffer(),
                                dstSlice.value().offset +
                                    std::uint64_t{slot} * vertexCount * kVertexStride,
                                vertexData.data(), std::uint64_t{vertexCount} * kVertexStride);
                            !staged) {
                            log::error("Posed-vertex seeding failed: {}", staged.error().message);
                            return 1;
                        }
                    }
                    AnimatedMeshEntry entry;
                    entry.objectIndex = static_cast<std::uint32_t>(geometry.size() - 1);
                    entry.vertexCount = vertexCount;
                    entry.srcVertex =
                        static_cast<std::uint32_t>(vertexSlice.value().offset / kVertexStride);
                    entry.dstVertexBase =
                        static_cast<std::uint32_t>(dstSlice.value().offset / kVertexStride);
                    entry.sourceNode = mesh.sourceNode;
                    entry.modelIndex = modelIndex;
                    if (mesh.skinned) {
                        entry.skinVertexOffset =
                            static_cast<std::uint32_t>(skinVertexData.size());
                        for (std::uint32_t v = 0; v < vertexCount; ++v) {
                            PackedSkinVertex packed;
                            packed.joints01 = mesh.joints[v * 4] |
                                              (std::uint32_t{mesh.joints[v * 4 + 1]} << 16);
                            packed.joints23 = mesh.joints[v * 4 + 2] |
                                              (std::uint32_t{mesh.joints[v * 4 + 3]} << 16);
                            for (int k = 0; k < 4; ++k) {
                                packed.weights[k] = mesh.weights[v * 4 + k];
                            }
                            skinVertexData.push_back(packed);
                        }
                    }
                    if (!mesh.morphTargets.empty()) {
                        entry.morphBase =
                            static_cast<std::uint32_t>(morphDeltaData.size() / 6);
                        entry.morphTargetCount =
                            static_cast<std::uint32_t>(mesh.morphTargets.size());
                        entry.morphWeightOffset =
                            static_cast<std::uint32_t>(morphWeightsCpu.size());
                        for (const auto& target : mesh.morphTargets) {
                            for (std::uint32_t v = 0; v < vertexCount; ++v) {
                                for (int c = 0; c < 3; ++c) {
                                    morphDeltaData.push_back(
                                        v * 3 + c < target.positionDeltas.size()
                                            ? target.positionDeltas[v * 3 + c]
                                            : 0.0f);
                                }
                                for (int c = 0; c < 3; ++c) {
                                    morphDeltaData.push_back(
                                        v * 3 + c < target.normalDeltas.size()
                                            ? target.normalDeltas[v * 3 + c]
                                            : 0.0f);
                                }
                            }
                        }
                        morphWeightsCpu.insert(morphWeightsCpu.end(), mesh.morphWeights.begin(),
                                               mesh.morphWeights.end());
                    }
                    animatedMeshes.push_back(entry);
                }
            }
        }
        // One indirect entry per mesh, addressing its slices by offset.
        // instanceCount is the milestone-7 load/unload toggle; firstInstance
        // becomes the object-SSBO index in step 3.
        // Reserve to full capacity so batch.cpuDraws (Direct mode) never
        // dangles when runtime loads append entries.
        templateCapacity = static_cast<std::uint32_t>(geometry.size()) + kRuntimeObjectCapacity;
        draws.reserve(templateCapacity);
        for (std::size_t i = 0; i < geometry.size(); ++i) {
            const GeometryLocation& location = geometry[i];
            draws.push_back({
                .indexCount = location.indexCount,
                .instanceCount = 1,
                .firstIndex = static_cast<std::uint32_t>(location.indices.offset /
                                                         sizeof(std::uint32_t)),
                .vertexOffset = static_cast<std::int32_t>(location.vertices.offset / kVertexStride),
                .firstInstance = static_cast<std::uint32_t>(i),
            });
        }
        // Host-visible with one region per frame in flight: the CPU rewrites
        // the current slot's instanceCounts (the milestone-7 load/unload
        // toggle, now driven by runtime load/unload messages) while the
        // other slot's region is in flight. Regions are CAPACITY-sized:
        // runtime loads append entries without moving the slot bases.
        const std::uint64_t indirectRegion =
            std::uint64_t{templateCapacity} * sizeof(gpu::DrawIndexedIndirect);
        auto indirectResult = gpu::Buffer::create(
            *device, {
                         .size = indirectRegion * gpu::FrameRenderer::kFramesInFlight,
                         // Storage too: the cull pass reads these entries as
                         // its draw templates (descriptor binding 3).
                         .usage = gpu::kUsageIndirect | gpu::kUsageStorage,
                         .location = gpu::MemoryLocation::HostVisible,
                     });
        if (!indirectResult) {
            log::error("Indirect buffer creation failed: {}", indirectResult.error().message);
            return 1;
        }
        indirectBuffer = std::move(indirectResult).value();
        // Each slot's template region points animated meshes at that slot's
        // posed-vertex copy; everything else is identical across slots.
        for (std::uint32_t slot = 0; slot < gpu::FrameRenderer::kFramesInFlight; ++slot) {
            std::vector<gpu::DrawIndexedIndirect> slotDraws = draws;
            for (const AnimatedMeshEntry& entry : animatedMeshes) {
                slotDraws[entry.objectIndex].vertexOffset =
                    static_cast<std::int32_t>(entry.dstVertexBase + slot * entry.vertexCount);
            }
            std::memcpy(static_cast<std::byte*>(indirectBuffer->mapped()) + slot * indirectRegion,
                        slotDraws.data(), slotDraws.size() * sizeof(gpu::DrawIndexedIndirect));
        }

        // Cull-pass output: the compacted indirect list the GPU builds each
        // frame (per-slot regions like the templates).
        auto compactedResult = gpu::Buffer::create(
            *device, {
                         .size = indirectRegion * gpu::FrameRenderer::kFramesInFlight,
                         .usage = gpu::kUsageIndirect | gpu::kUsageStorage,
                         .location = gpu::MemoryLocation::DeviceLocal,
                     });
        if (!compactedResult) {
            log::error("Compacted buffer creation failed: {}", compactedResult.error().message);
            return 1;
        }
        compactedBuffer = std::move(compactedResult).value();

        // Second cull-pass output: the frustum-culled list the main scene
        // pass draws (the compacted list above keeps every live entry for
        // the shadow passes — casters outside the view must still cast).
        auto culledResult = gpu::Buffer::create(
            *device, {
                         .size = indirectRegion * gpu::FrameRenderer::kFramesInFlight,
                         .usage = gpu::kUsageIndirect | gpu::kUsageStorage,
                         .location = gpu::MemoryLocation::DeviceLocal,
                     });
        if (!culledResult) {
            log::error("Culled buffer creation failed: {}", culledResult.error().message);
            return 1;
        }
        culledBuffer = std::move(culledResult).value();

        // Third cull-pass output: the frustum-culled TRANSPARENT list the
        // blend pass draws after the opaques (binding 24).
        auto transparentResult = gpu::Buffer::create(
            *device, {
                         .size = indirectRegion * gpu::FrameRenderer::kFramesInFlight,
                         .usage = gpu::kUsageIndirect | gpu::kUsageStorage,
                         .location = gpu::MemoryLocation::DeviceLocal,
                     });
        if (!transparentResult) {
            log::error("Transparent buffer creation failed: {}",
                       transparentResult.error().message);
            return 1;
        }
        transparentBuffer = std::move(transparentResult).value();

        // Draw-count buffer for IndirectCount mode: written by the cull
        // pass (zeroed via fill, incremented by the shader) and read by
        // vkCmdDrawIndexedIndirectCount. Eight counters per frame slot
        // (must match kCountStride in cull.hlsl): [0] the shadow passes'
        // visibility-only stream, [1] the scene pass's frustum-culled
        // opaque stream, [2] the scratch-row allocator for per-instance
        // culling, [3] the blend pass's transparent stream, [4]/[5] the
        // indices emitted to the opaque/transparent streams (post-LOD —
        // the triangle stat, /3), [6][7] spare. Host-visible so the CPU
        // can report them (read after the slot's fence, i.e.
        // kFramesInFlight frames late — stats, never a sync point).
        auto countResult = gpu::Buffer::create(
            *device, {
                         .size = 8 * sizeof(std::uint32_t) * gpu::FrameRenderer::kFramesInFlight,
                         .usage = gpu::kUsageIndirect | gpu::kUsageStorage |
                                  gpu::kUsageTransferDst,
                         .location = gpu::MemoryLocation::HostVisible,
                     });
        if (!countResult) {
            log::error("Count buffer creation failed: {}", countResult.error().message);
            return 1;
        }
        countBuffer = std::move(countResult).value();

        // Camera matrices, one region per frame slot: rewritten by the CPU
        // every frame (after waitFrameSlot), read by the vertex shader via
        // the slot index push constant. This is what keeps static command
        // buffers valid while the camera moves. Six extra regions follow
        // the per-frame ones: the probe capture pass pushes slot =
        // kFramesInFlight + face, so the same shaders render cube faces.
        auto cameraResult = gpu::Buffer::create(
            *device, {
                         .size = sizeof(CameraData) * kCameraRegions,
                         .usage = gpu::kUsageStorage,
                         .location = gpu::MemoryLocation::HostVisible,
                     });
        if (!cameraResult) {
            log::error("Camera buffer creation failed: {}", cameraResult.error().message);
            return 1;
        }
        cameraBuffer = std::move(cameraResult).value();

        // Per-object transforms: per-camera-slot regions (frame slots +
        // probe capture faces), all seeded identity — scene geometry is
        // world-baked, only runtime-spawned models carry real matrices.
        if (templateCapacity > kTransformCapacity) {
            log::error("Scene needs {} transform rows, capacity {}", templateCapacity,
                       kTransformCapacity);
            return 1;
        }
        auto transformResult = gpu::Buffer::create(
            *device, {
                         .size = std::uint64_t{kTransformCapacity} * sizeof(math::Mat4) *
                                 kCameraRegions,
                         .usage = gpu::kUsageStorage,
                         .location = gpu::MemoryLocation::HostVisible,
                     });
        if (!transformResult) {
            log::error("Transform buffer creation failed: {}", transformResult.error().message);
            return 1;
        }
        transformBuffer = std::move(transformResult).value();
        {
            auto* mats = static_cast<math::Mat4*>(transformBuffer->mapped());
            for (std::size_t i = 0; i < std::size_t{kTransformCapacity} * kCameraRegions; ++i) {
                mats[i] = kIdentityMat4;
            }
        }
        // Scene draws ride the identity prefix; runtime instances allocate
        // rows past it (freed indices recycle before the vector grows).
        objectTransforms.assign(geometry.size(), kIdentityMat4);

        // Instance rows: identity prefix for the scene's draws, runtime
        // blocks above (the canonical region — see the constant's
        // comment), followed by one GPU-written scratch region per frame
        // slot where the cull pass compacts the surviving rows of
        // partially visible instanced draws.
        auto instanceRowResult = gpu::Buffer::create(
            *device, {
                         .size = std::uint64_t{kInstanceRowCapacity} * sizeof(InstanceRow) *
                                 (1 + gpu::FrameRenderer::kFramesInFlight),
                         .usage = gpu::kUsageStorage,
                         .location = gpu::MemoryLocation::HostVisible,
                     });
        if (!instanceRowResult) {
            log::error("Instance row buffer creation failed: {}",
                       instanceRowResult.error().message);
            return 1;
        }
        instanceRowBuffer = std::move(instanceRowResult).value();
        if (geometry.size() > kInstanceRowCapacity) {
            log::error("Scene needs {} instance rows, capacity {}", geometry.size(),
                       kInstanceRowCapacity);
            return 1;
        }
        instanceRows.assign(kInstanceRowCapacity, InstanceRow{});
        for (std::uint32_t i = 0; i < geometry.size(); ++i) {
            instanceRows[i] = {.objectIndex = i, .transformIndex = i};
        }
        std::memcpy(instanceRowBuffer->mapped(), instanceRows.data(),
                    instanceRows.size() * sizeof(InstanceRow));

        // Per-object bounds for the cull shader's frustum test, one region
        // per frame slot (the CPU rewrites the current slot's copy every
        // frame — runtime instances move). Seed every region now.
        auto boundsResult = gpu::Buffer::create(
            *device, {
                         .size = std::uint64_t{templateCapacity} * sizeof(ObjectBounds) *
                                 gpu::FrameRenderer::kFramesInFlight,
                         .usage = gpu::kUsageStorage,
                         .location = gpu::MemoryLocation::HostVisible,
                     });
        if (!boundsResult) {
            log::error("Bounds buffer creation failed: {}", boundsResult.error().message);
            return 1;
        }
        boundsBuffer = std::move(boundsResult).value();
        for (std::uint32_t s = 0; s < gpu::FrameRenderer::kFramesInFlight; ++s) {
            std::memcpy(static_cast<std::byte*>(boundsBuffer->mapped()) +
                            std::uint64_t{s} * templateCapacity * sizeof(ObjectBounds),
                        objectBounds.data(), objectBounds.size() * sizeof(ObjectBounds));
        }

        // Per-entry LOD tables: static after load, so ONE global region —
        // no per-slot copies. Zero-fill the runtime headroom (and any
        // chainless entries): lodCount 0 reads as "template stands".
        auto lodTableResult = gpu::Buffer::create(
            *device, {
                         .size = std::uint64_t{templateCapacity} * sizeof(MeshLodTable),
                         .usage = gpu::kUsageStorage,
                         .location = gpu::MemoryLocation::HostVisible,
                     });
        if (!lodTableResult) {
            log::error("LOD table buffer creation failed: {}", lodTableResult.error().message);
            return 1;
        }
        meshLodBuffer = std::move(lodTableResult).value();
        std::memset(meshLodBuffer->mapped(), 0, meshLodBuffer->size());
        std::memcpy(meshLodBuffer->mapped(), meshLodTables.data(),
                    meshLodTables.size() * sizeof(MeshLodTable));
        if (lodMeshCount > 0) {
            log::info("LOD chains: {} of {} meshes, {:.1f} MiB of simplified indices",
                      lodMeshCount, geometry.size(),
                      static_cast<double>(lodExtraIndices * sizeof(std::uint32_t)) /
                          (1024.0 * 1024.0));
        }

        // Occlusion visibility: device-local (the proxy passes hammer it
        // with fragment stores), seeded all-ones so the first frames draw
        // everything until real proxy results exist. Each slot region =
        // templateCapacity ENTRY slots + kInstanceRowCapacity per-
        // INSTANCE slots (runtime instanced models — cull.hlsl and
        // proxy.hlsl hardcode the same split).
        const std::uint64_t visibilitySlots =
            std::uint64_t{templateCapacity} + kInstanceRowCapacity;
        auto visibilityResult = gpu::Buffer::create(
            *device, {
                         .size = visibilitySlots * sizeof(std::uint32_t) *
                                 gpu::FrameRenderer::kFramesInFlight,
                         .usage = gpu::kUsageStorage | gpu::kUsageTransferDst,
                         .location = gpu::MemoryLocation::DeviceLocal,
                     });
        if (!visibilityResult) {
            log::error("Visibility buffer creation failed: {}",
                       visibilityResult.error().message);
            return 1;
        }
        visibilityBuffer = std::move(visibilityResult).value();
        {
            const std::vector<std::uint32_t> seed(
                visibilitySlots * gpu::FrameRenderer::kFramesInFlight, 1u);
            if (auto staged = transfer->stage(*visibilityBuffer, 0, seed.data(),
                                              seed.size() * sizeof(std::uint32_t));
                !staged) {
                log::error("Visibility seeding failed: {}", staged.error().message);
                return 1;
            }
        }

        // Canonical-row -> entry map for the per-instance proxy pass:
        // host-visible (rewritten on add/remove/reclaim, same benign
        // in-flight races as the instance rows themselves), all-dead
        // until runtime models land.
        auto rowEntryResult = gpu::Buffer::create(
            *device, {
                         .size = std::uint64_t{kInstanceRowCapacity} * sizeof(std::uint32_t),
                         .usage = gpu::kUsageStorage,
                         .location = gpu::MemoryLocation::HostVisible,
                     });
        if (!rowEntryResult) {
            log::error("Row-entry buffer creation failed: {}", rowEntryResult.error().message);
            return 1;
        }
        rowEntryBuffer = std::move(rowEntryResult).value();
        std::memset(rowEntryBuffer->mapped(), 0xff,
                    std::size_t{kInstanceRowCapacity} * sizeof(std::uint32_t));

        // Refined-OBB table: header (claim counter) + one row per entry,
        // device-local and GPU-written only. Zero-seeded: counter 0, every
        // row unready — consumers keep the AABBs until the baked refine
        // dispatch fits a tighter box. ONE global region (rows converge
        // to load-time constants; the recorded barriers order access).
        auto obbResult = gpu::Buffer::create(
            *device, {
                         .size = kObbHeaderBytes + std::uint64_t{templateCapacity} * kObbRowBytes,
                         .usage = gpu::kUsageStorage | gpu::kUsageTransferDst,
                         .location = gpu::MemoryLocation::DeviceLocal,
                     });
        if (!obbResult) {
            log::error("OBB buffer creation failed: {}", obbResult.error().message);
            return 1;
        }
        obbBuffer = std::move(obbResult).value();
        {
            const std::vector<std::uint8_t> zeros(obbBuffer->size(), 0);
            if (auto staged = transfer->stage(*obbBuffer, 0, zeros.data(), zeros.size());
                !staged) {
                log::error("OBB buffer seeding failed: {}", staged.error().message);
                return 1;
            }
        }

        // Light data, same per-slot scheme (and capture regions) as the camera.
        auto lightResult = gpu::Buffer::create(
            *device, {
                         .size = sizeof(LightData) * kCameraRegions,
                         .usage = gpu::kUsageStorage,
                         .location = gpu::MemoryLocation::HostVisible,
                     });
        if (!lightResult) {
            log::error("Light buffer creation failed: {}", lightResult.error().message);
            return 1;
        }
        lightBuffer = std::move(lightResult).value();

        for (std::uint32_t c = 0; c < kShadowCascades; ++c) {
            auto shadowResult = gpu::Image::create(
                *device, {
                             .width = kShadowMapSize,
                             .height = kShadowMapSize,
                             .format = gpu::kFormatD32Sfloat,
                             .usage = gpu::kImageUsageDepthAttachment | gpu::kImageUsageSampled,
                             .depth = true,
                         });
            if (!shadowResult) {
                log::error("Shadow map creation failed: {}", shadowResult.error().message);
                return 1;
            }
            shadowMaps.push_back(std::move(shadowResult).value());
        }

        auto objectResult = gpu::Buffer::create(
            *device, {
                         // Capacity headroom: runtime loads stage new rows
                         // in place; the descriptor is written once, full-size.
                         .size = (objectData.size() + kRuntimeObjectCapacity) * sizeof(ObjectData),
                         .usage = gpu::kUsageStorage | gpu::kUsageTransferDst,
                         .location = gpu::MemoryLocation::DeviceLocal,
                         .sharedWithTransferQueue = true,
                     });
        if (!objectResult) {
            log::error("Object buffer creation failed: {}", objectResult.error().message);
            return 1;
        }
        objectBuffer = std::move(objectResult).value();
        if (auto staged = transfer->stage(*objectBuffer, 0, objectData.data(),
                                          objectData.size() * sizeof(ObjectData));
            !staged) {
            log::error("Object staging failed: {}", staged.error().message);
            return 1;
        }

        // Triangle lookup for the traced primary pass: where each object's
        // indices/vertices sit in the pool, in element units.
        if (rtSupported) {
            std::vector<GeometryInfo> geometryInfo;
            geometryInfo.reserve(geometry.size());
            for (const GeometryLocation& location : geometry) {
                geometryInfo.push_back({
                    .firstIndex = static_cast<std::uint32_t>(location.indices.offset /
                                                             sizeof(std::uint32_t)),
                    .vertexOffset =
                        static_cast<std::uint32_t>(location.vertices.offset / kVertexStride),
                });
            }
            for (const AnimatedMeshEntry& entry : animatedMeshes) {
                geometryInfo[entry.objectIndex].vertexOffset = entry.dstVertexBase;
                geometryInfo[entry.objectIndex].slotStride = entry.vertexCount;
            }
            auto infoResult = gpu::Buffer::create(
                *device, {
                             // Same runtime headroom as the object buffer:
                             // integrateResource stages rows for runtime
                             // meshes in place (recycled or appended).
                             .size = (geometryInfo.size() + kRuntimeObjectCapacity) *
                                     sizeof(GeometryInfo),
                             .usage = gpu::kUsageStorage | gpu::kUsageTransferDst,
                             .location = gpu::MemoryLocation::DeviceLocal,
                             .sharedWithTransferQueue = true,
                         });
            if (!infoResult) {
                log::error("Geometry info buffer creation failed: {}", infoResult.error().message);
                return 1;
            }
            geometryInfoBuffer = std::move(infoResult).value();
            if (auto staged = transfer->stage(*geometryInfoBuffer, 0, geometryInfo.data(),
                                              geometryInfo.size() * sizeof(GeometryInfo));
                !staged) {
                log::error("Geometry info staging failed: {}", staged.error().message);
                return 1;
            }
            // Hit->object remap for runtime models (binding 32): one uint
            // per runtime mesh row, allocated in contiguous per-resource
            // blocks that TLAS instances point at via customIndex.
            auto remapResult = gpu::Buffer::create(
                *device, {
                             .size = kRuntimeObjectCapacity * sizeof(std::uint32_t),
                             .usage = gpu::kUsageStorage | gpu::kUsageTransferDst,
                             .location = gpu::MemoryLocation::DeviceLocal,
                             .sharedWithTransferQueue = true,
                         });
            if (!remapResult) {
                log::error("RT remap buffer creation failed: {}", remapResult.error().message);
                return 1;
            }
            rtRemapBuffer = std::move(remapResult).value();
        }

        // Animation state: playback per animated model plus the GPU-side
        // inputs (static skin/morph data staged, per-slot joint matrices
        // and weights CPU-written each frame).
        if (!animatedMeshes.empty()) {
            for (std::size_t modelIndex = 0; modelIndex < scene->models.size(); ++modelIndex) {
                const auto& model = scene->models[modelIndex];
                if (model.data.skeleton.empty()) {
                    continue;
                }
                AnimatedModelState state;
                state.data = &model.data;
                state.modelIndex = modelIndex;
                state.modelMatrix = transformMatrix(model.desc.transform);
                state.jointBase = totalJoints;
                totalJoints +=
                    static_cast<std::uint32_t>(model.data.skeleton.jointNodes.size());
                animatedStates.push_back(std::move(state));
            }
            jointMatricesCpu.resize(totalJoints);
            morphWeightsFrame = morphWeightsCpu;

            auto createStaged = [&](std::vector<std::unique_ptr<gpu::Buffer>>::value_type& out,
                                    const void* bytes, std::uint64_t size, const char* what) {
                if (size == 0) {
                    return true;
                }
                auto result = gpu::Buffer::create(
                    *device, {
                                 .size = size,
                                 .usage = gpu::kUsageStorage | gpu::kUsageTransferDst,
                                 .location = gpu::MemoryLocation::DeviceLocal,
                                 .sharedWithTransferQueue = true,
                             });
                if (!result || !transfer->stage(*result.value(), 0, bytes, size)) {
                    log::error("{} buffer failed", what);
                    return false;
                }
                out = std::move(result).value();
                return true;
            };
            if (!createStaged(skinVertexBuffer, skinVertexData.data(),
                              skinVertexData.size() * sizeof(PackedSkinVertex), "Skin vertex") ||
                !createStaged(morphDeltaBuffer, morphDeltaData.data(),
                              morphDeltaData.size() * sizeof(float), "Morph delta")) {
                return 1;
            }
            auto createPerSlot = [&](std::unique_ptr<gpu::Buffer>& out, std::uint64_t regionSize,
                                     const char* what) {
                if (regionSize == 0) {
                    return true;
                }
                auto result = gpu::Buffer::create(
                    *device, {
                                 .size = regionSize * gpu::FrameRenderer::kFramesInFlight,
                                 .usage = gpu::kUsageStorage,
                                 .location = gpu::MemoryLocation::HostVisible,
                             });
                if (!result) {
                    log::error("{} buffer failed: {}", what, result.error().message);
                    return false;
                }
                out = std::move(result).value();
                return true;
            };
            if (!createPerSlot(jointBuffer, totalJoints * sizeof(math::Mat4), "Joint matrix") ||
                !createPerSlot(morphWeightBuffer, morphWeightsFrame.size() * sizeof(float),
                               "Morph weight")) {
                return 1;
            }
            log::info("Animation ready: {} animated meshes, {} joints, {} morph weights",
                      animatedMeshes.size(), totalJoints, morphWeightsFrame.size());
        }

        {
            REND_PROFILE_ZONE("GeometryUpload");
            if (auto flushed = transfer->flush(); !flushed) {
                log::error("Geometry upload failed: {}", flushed.error().message);
                return 1;
            }
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        // Ray-traced shadows need the scene in a BVH: one BLAS over every
        // mesh range in the pool, one identity instance in the TLAS
        // (geometry is world-space baked). Load-time build; failures just
        // leave the shadow-map path.
        if (rtSupported) {
            const auto rtStart = std::chrono::steady_clock::now();
            std::vector<gpu::AccelerationStructure::TriangleGeometry> triangles;
            triangles.reserve(geometry.size());
            for (std::size_t i = 0; i < geometry.size(); ++i) {
                const GeometryLocation& location = geometry[i];
                triangles.push_back({
                    .buffer = &geometryPool->buffer(),
                    .vertexOffset = location.vertices.offset,
                    .vertexStride = kVertexStride,
                    .vertexCount =
                        static_cast<std::uint32_t>(location.vertices.size / kVertexStride),
                    .indexOffset = location.indices.offset,
                    .indexCount = location.indexCount,
                    // Masked/blended surfaces stay non-opaque so traced
                    // rays can alpha-test or march through them.
                    .opaque = (objectData[i].flags &
                               (kObjectAlphaMasked | kObjectTransparent)) == 0,
                });
            }
            // Animated meshes need per-frame refits: build updatable and
            // precompute each slot's geometry list — identical to the build
            // except animated entries read that slot's posed region.
            const bool animatedBvh = !animatedMeshes.empty();
            auto blasResult =
                gpu::AccelerationStructure::buildBottomLevel(*device, triangles, animatedBvh);
            if (blasResult) {
                blas = std::move(blasResult).value();
                // Dynamic TLAS: instance 0 = the world-baked scene BLAS
                // (sentinel customIndex), the rest is capacity for runtime
                // model instances written per frame + rebuilt in place.
                const gpu::AccelerationStructure::Instance blasInstance{
                    .blas = blas.get(), .customIndex = kRtSceneCustomIndex};
                auto tlasResult = gpu::AccelerationStructure::buildTopLevelDynamic(
                    *device, {&blasInstance, 1}, kRtInstanceCapacity,
                    gpu::FrameRenderer::kFramesInFlight);
                if (tlasResult) {
                    tlas = std::move(tlasResult).value();
                    rtReady = true;
                    if (animatedBvh) {
                        for (std::uint32_t slot = 0; slot < gpu::FrameRenderer::kFramesInFlight;
                             ++slot) {
                            auto slotTriangles = triangles;
                            for (const AnimatedMeshEntry& entry : animatedMeshes) {
                                slotTriangles[entry.objectIndex].vertexOffset =
                                    (std::uint64_t{entry.dstVertexBase} +
                                     std::uint64_t{slot} * entry.vertexCount) *
                                    kVertexStride;
                            }
                            refitGeometries.push_back(std::move(slotTriangles));
                        }
                    }
                    const auto rtMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          std::chrono::steady_clock::now() - rtStart)
                                          .count();
                    log::info("Ray-tracing BVH ready in {} ms", rtMs);
                } else {
                    log::warn("TLAS build failed: {}", tlasResult.error().message);
                }
            } else {
                log::warn("BLAS build failed: {}", blasResult.error().message);
            }
        }

        log::info("Geometry uploaded in {} ms: {} slices, {:.1f} MiB of {} MiB used ({} queue)",
                  ms, geometryPool->allocationCount(),
                  static_cast<double>(geometryPool->usedBytes()) / (1024.0 * 1024.0),
                  geometryPool->capacity() / (1024 * 1024),
                  device->hasDedicatedTransfer() ? "dedicated transfer" : "graphics");

        // Bindless table: decode every referenced base-color texture through
        // assetio, upload with full mip chains, and point the object SSBO
        // rows at their slots.
        const auto texStart = std::chrono::steady_clock::now();
        auto tableResult = gpu::DescriptorTable::create(*device, 1024);
        if (!tableResult) {
            log::error("Descriptor table creation failed: {}", tableResult.error().message);
            return 1;
        }
        descriptorTable = std::move(tableResult).value();
        descriptorTable->writeObjectBuffer(objectBuffer->handle(), objectBuffer->size());
        descriptorTable->writeStorageBuffer(19, transformBuffer->handle(),
                                            transformBuffer->size());
        descriptorTable->writeStorageBuffer(20, instanceRowBuffer->handle(),
                                            instanceRowBuffer->size());
        descriptorTable->writeStorageBuffer(21, culledBuffer->handle(), culledBuffer->size());
        descriptorTable->writeStorageBuffer(24, transparentBuffer->handle(),
                                            transparentBuffer->size());
        descriptorTable->writeStorageBuffer(22, boundsBuffer->handle(), boundsBuffer->size());
        descriptorTable->writeStorageBuffer(25, meshLodBuffer->handle(), meshLodBuffer->size());
        descriptorTable->writeStorageBuffer(33, obbBuffer->handle(), obbBuffer->size());
        descriptorTable->writeStorageBuffer(34, rowEntryBuffer->handle(),
                                            rowEntryBuffer->size());
        descriptorTable->writeStorageBuffer(26, visibilityBuffer->handle(),
                                            visibilityBuffer->size());
        // The rows buffer again, writable for the cull pass's scratch
        // regions (same VkBuffer, second binding — no aliasing hazard,
        // canonical and scratch ranges are disjoint).
        descriptorTable->writeStorageBuffer(23, instanceRowBuffer->handle(),
                                            instanceRowBuffer->size());
        descriptorTable->writeStorageBuffer(3, indirectBuffer->handle(), indirectBuffer->size());
        descriptorTable->writeStorageBuffer(4, compactedBuffer->handle(),
                                            compactedBuffer->size());
        descriptorTable->writeStorageBuffer(5, countBuffer->handle(), countBuffer->size());
        descriptorTable->writeStorageBuffer(6, cameraBuffer->handle(), cameraBuffer->size());
        descriptorTable->writeStorageBuffer(7, lightBuffer->handle(), lightBuffer->size());
        // The pool's compute view (binding 13) is always bound: the skin
        // pass writes posed vertices through it AND the OBB refine pass
        // reads static vertices from it — static scenes need it too.
        descriptorTable->writeStorageBuffer(13, geometryPool->buffer().handle(),
                                            geometryPool->buffer().size());
        if (!animatedMeshes.empty()) {
            if (skinVertexBuffer) {
                descriptorTable->writeStorageBuffer(14, skinVertexBuffer->handle(),
                                                    skinVertexBuffer->size());
            }
            if (jointBuffer) {
                descriptorTable->writeStorageBuffer(15, jointBuffer->handle(),
                                                    jointBuffer->size());
            }
            if (morphDeltaBuffer) {
                descriptorTable->writeStorageBuffer(16, morphDeltaBuffer->handle(),
                                                    morphDeltaBuffer->size());
            }
            if (morphWeightBuffer) {
                descriptorTable->writeStorageBuffer(17, morphWeightBuffer->handle(),
                                                    morphWeightBuffer->size());
            }
        }
        for (std::uint32_t c = 0; c < kShadowCascades; ++c) {
            descriptorTable->writeShadowMap(c, shadowMaps[c]->view());
        }
        if (rtReady) {
            descriptorTable->writeAccelerationStructure(tlas->handle());
            // Traced-primary triangle fetch: the pool's raw bytes plus the
            // per-object index/vertex offsets.
            descriptorTable->writeStorageBuffer(11, geometryPool->buffer().handle(),
                                                geometryPool->buffer().size());
            descriptorTable->writeStorageBuffer(12, geometryInfoBuffer->handle(),
                                                geometryInfoBuffer->size());
            descriptorTable->writeStorageBuffer(32, rtRemapBuffer->handle(),
                                                rtRemapBuffer->size());
        }

        auto uploaderResult = gpu::TextureUploader::create(*device);
        if (!uploaderResult) {
            log::error("Texture uploader creation failed: {}", uploaderResult.error().message);
            return 1;
        }
        uploader = std::move(uploaderResult).value();

        // Nonblocking texture streaming: the white fallback uploads now and
        // EVERY registered slot starts pointing at it, so any material
        // samples neutrally until its real texture lands. Decode workers
        // run while the frame loop renders; the loop's pump uploads and
        // re-points slots (see SceneTextureStream at main scope).
        {
            const std::uint8_t white[4] = {255, 255, 255, 255};
            auto whiteResult = uploader->upload(1, 1, white);
            if (!whiteResult) {
                log::error("White fallback upload failed: {}", whiteResult.error().message);
                return 1;
            }
            textures.resize(texturePaths.size());
            textures[0] = std::move(whiteResult).value();
            for (std::size_t i = 0; i < texturePaths.size(); ++i) {
                descriptorTable->writeTexture(static_cast<std::uint32_t>(i),
                                              textures[0]->view());
            }
        }
        texStream.total = texturePaths.size() - 1;
        texStream.active = texStream.total > 0;
        texStream.start = texStart;
        texStream.lastFlush = std::chrono::steady_clock::now();
        waitForTextures =
            texStream.active && scene->loading == assetio::SceneLoadingMode::Wait;
        if (texStream.active) {
            const std::size_t workerCount = std::min<std::size_t>(
                std::max(1u, std::thread::hardware_concurrency()), texStream.total);
            texStream.workers.reserve(workerCount);
            for (std::size_t w = 0; w < workerCount; ++w) {
                texStream.workers.emplace_back([&] {
                    for (std::size_t i = texStream.next.fetch_add(1);
                         i < texturePaths.size() &&
                         !texStream.quit.load(std::memory_order_relaxed);
                         i = texStream.next.fetch_add(1)) {
                        REND_PROFILE_ZONE("TextureDecode");
                        auto result = assetio::loadTexture(texturePaths[i].path);
                        std::lock_guard lock(texStream.mutex);
                        texStream.ready.emplace_back(i, std::move(result));
                    }
                });
            }
            log::info("Texture streaming started: {} textures on {} workers ({} mode)",
                      texStream.total, workerCount,
                      waitForTextures ? "wait" : "streaming");
        } else {
            log::info("Textures ready in 0 ms: 1 images (white fallback only)");
        }

        // Static <Model> bounds are final here — snapshot them for the
        // runtime-instance cascade-fit recompute and the probe position.
        staticSceneMin = sceneMin;
        staticSceneMax = sceneMax;

        // Reflection probe capture: render the scene's draw stream into a
        // small cubemap once — the raster tier reflective objects sample.
        // Static by nature: animated meshes bake at the bind pose, lighting
        // never re-captures, no parallax correction (v1 gaps by design;
        // ray traced reflections are the exact tier above). DEFERRED until
        // the texture stream completes so the capture samples the real
        // textures — the frame loop's pump invokes this once.
        // Snapshot the static-scene draw count now: runtime loads only
        // append entries, and those must not reach either capture — the
        // capture-face transform regions are seeded identity and never
        // rewritten, so a runtime instance resident at capture time would
        // bake into the cubes untransformed at the origin.
        capturePendingProbe = [&, sceneDrawCount = static_cast<std::uint32_t>(draws.size())] {
            const auto probeStart = std::chrono::steady_clock::now();
            // Probe position: first <ReflectionProbe> in the scene XML,
            // else the STATIC scene AABB's center (runtime spawns are
            // excluded from the capture, so they don't steer it either).
            math::Vec3 probePos = vmul(vadd(staticSceneMin, staticSceneMax), 0.5f);
            if (!scene->reflectionProbes.empty()) {
                const auto& p = scene->reflectionProbes.front().position;
                probePos = {p[0], p[1], p[2]};
            }

            // Six capture camera regions after the per-frame ones: 90-degree
            // faces in the spec's cube texel basis (see kProbeFaceBases).
            // Shared with the point-shadow bake below, which re-centers
            // the same regions on each light before its own capture.
            const math::Mat4 proj = math::perspective(kPi * 0.5f, 1.0f, 0.05f, 300.0f);
            auto writeCaptureCameras = [&](const math::Vec3& pos) {
                for (std::uint32_t face = 0; face < kProbeFaces; ++face) {
                    const ProbeFaceBasis& basis = kProbeFaceBases[face];
                    CameraData faceCamera;
                    faceCamera.viewProj =
                        math::mul(proj, faceView(pos, basis.right, basis.up, basis.forward));
                    faceCamera.position = {pos.x, pos.y, pos.z, 0.001f};
                    faceCamera.rightAxis = {basis.right.x, basis.right.y, basis.right.z, 0.0f};
                    faceCamera.upAxis = {basis.up.x, basis.up.y, basis.up.z, 0.0f};
                    faceCamera.forwardAxis = {basis.forward.x, basis.forward.y,
                                              basis.forward.z, 0.0f};
                    std::memcpy(static_cast<std::byte*>(cameraBuffer->mapped()) +
                                    (gpu::FrameRenderer::kFramesInFlight + face) *
                                        sizeof(CameraData),
                                &faceCamera, sizeof(CameraData));
                }
            };
            writeCaptureCameras(probePos);
            for (std::uint32_t face = 0; face < kProbeFaces; ++face) {
                // Matching light regions: sun color/direction but no shadow
                // sampling (cascadeCount 0, no maps exist yet) and no
                // reflections — the capture sees reflective objects as
                // plain surfaces instead of sampling the probe being made.
                LightData faceLight;
                const SunControls captureSun =
                    SunControls::fromLight(SunControls::sceneSun(scene->lights));
                const math::Vec3 dir = captureSun.direction();
                faceLight.direction = {dir.x, dir.y, dir.z};
                faceLight.intensity = captureSun.intensity;
                faceLight.color = captureSun.color;
                faceLight.cascadeCount = 0;
                faceLight.rtShadows = 0;
                faceLight.reflections = kReflectionNone;
                std::memcpy(static_cast<std::byte*>(lightBuffer->mapped()) +
                                (gpu::FrameRenderer::kFramesInFlight + face) *
                                    sizeof(LightData),
                            &faceLight, sizeof(LightData));
            }

            // Dedicated pipeline over the probe color format: the standard
            // scene shaders, always the non-RT fragment variant (capture
            // neither traces nor samples shadow maps).
            const auto probeShaderDir = executableDirectory() / "data" / "shaders";
            auto probeVertResult =
                gpu::Shader::createFromFile(*device, probeShaderDir / "scene.vert.spv");
            auto probeFragResult =
                gpu::Shader::createFromFile(*device, probeShaderDir / "scene.frag.spv");
            if (probeVertResult && probeFragResult) {
                auto probeVert = std::move(probeVertResult).value();
                auto probeFrag = std::move(probeFragResult).value();
                auto probePipeResult = gpu::Pipeline::createGraphics(
                    *device,
                    {
                        .vertexShader = probeVert.get(),
                        .fragmentShader = probeFrag.get(),
                        .colorFormat = kProbeFormat,
                        .vertexStride = kVertexStride,
                        .vertexAttributes = {{0, gpu::kFormatR32G32B32Sfloat, 0},
                                             {1, gpu::kFormatR32G32B32Sfloat, 12},
                                             {2, gpu::kFormatR32G32Sfloat, 24}},
                        .depthFormat = gpu::kFormatD32Sfloat,
                        .pushConstantBytes = 2 * sizeof(std::uint32_t),
                        .descriptorLayout = descriptorTable->layout(),
                    });
                if (probePipeResult) {
                    auto probePipeline = std::move(probePipeResult).value();
                    auto probeResult = gpu::ProbeCapture::render(
                        *device, {
                                     .geometry = geometryPool->buffer().handle(),
                                     .draws = draws.data(),
                                     .drawCount = sceneDrawCount,
                                     .descriptors = descriptorTable->set(),
                                     .pipeline = probePipeline.get(),
                                     .format = kProbeFormat,
                                     .faceSize = kProbeFaceSize,
                                     .cameraSlotBase = gpu::FrameRenderer::kFramesInFlight,
                                 });
                    if (probeResult) {
                        probeImage = std::move(probeResult).value();
                        // Binding 18 is not update-after-bind: the CALLER
                        // idles the device around this invocation and
                        // invalidates static recordings after it.
                        descriptorTable->writeProbe(probeImage->view());
                        probeReady = true;
                        const auto probeMs =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - probeStart)
                                .count();
                        log::info("Reflection probe captured in {} ms: {}^2 x6 faces at "
                                  "({:.2f} {:.2f} {:.2f}), {} mips",
                                  probeMs, kProbeFaceSize, probePos.x, probePos.y, probePos.z,
                                  probeImage->mipLevels());
                    } else {
                        log::warn("Reflection probe capture failed: {}",
                                  probeResult.error().message);
                    }
                } else {
                    log::warn("Probe pipeline unavailable: {}", probePipeResult.error().message);
                }
            } else {
                log::warn("Probe shaders unavailable: {}",
                          (!probeVertResult ? probeVertResult : probeFragResult)
                              .error()
                              .message);
            }

            // Point-light shadow bake: for each shadow-casting scene
            // light, the probe-capture machinery renders the scene's
            // distance-from-light into a small R32F cube (level 0 only,
            // cleared to "no occluder"), re-centering the capture camera
            // regions on the light. Static scene only — runtime models
            // never occlude lamp light (they're spawned after this).
            bool wantPointShadows = false;
            for (const PointLight& pl : pointLights) {
                wantPointShadows |= pl.params[0] > 0.0f && pl.colorIntensity[3] > 0.0f;
            }
            if (wantPointShadows) {
                const auto bakeStart = std::chrono::steady_clock::now();
                auto psVertResult =
                    gpu::Shader::createFromFile(*device, probeShaderDir / "scene.vert.spv");
                auto psFragResult = gpu::Shader::createFromFile(
                    *device, probeShaderDir / "point_shadow.frag.spv");
                std::unique_ptr<gpu::Pipeline> bakePipeline;
                if (psVertResult && psFragResult) {
                    auto psVert = std::move(psVertResult).value();
                    auto psFrag = std::move(psFragResult).value();
                    auto bakeResult = gpu::Pipeline::createGraphics(
                        *device,
                        {
                            .vertexShader = psVert.get(),
                            .fragmentShader = psFrag.get(),
                            .colorFormat = gpu::kFormatR32Sfloat,
                            .vertexStride = kVertexStride,
                            .vertexAttributes = {{0, gpu::kFormatR32G32B32Sfloat, 0},
                                                 {1, gpu::kFormatR32G32B32Sfloat, 12},
                                                 {2, gpu::kFormatR32G32Sfloat, 24}},
                            .depthFormat = gpu::kFormatD32Sfloat,
                            .pushConstantBytes = 2 * sizeof(std::uint32_t),
                            .descriptorLayout = descriptorTable->layout(),
                        });
                    if (bakeResult) {
                        bakePipeline = std::move(bakeResult).value();
                    } else {
                        log::warn("Point-shadow pipeline unavailable: {}",
                                  bakeResult.error().message);
                    }
                } else {
                    log::warn("Point-shadow shader unavailable: {}",
                              (!psVertResult ? psVertResult : psFragResult).error().message);
                }
                std::uint32_t baked = 0;
                if (bakePipeline) {
                    pointShadowImages.resize(kMaxPointLights);
                    for (std::uint32_t i = 0; i < kMaxPointLights; ++i) {
                        PointLight& pl = pointLights[i];
                        if (pl.params[0] <= 0.0f || pl.colorIntensity[3] <= 0.0f) {
                            continue;
                        }
                        writeCaptureCameras({pl.positionRadius[0], pl.positionRadius[1],
                                             pl.positionRadius[2]});
                        auto cubeResult = gpu::ProbeCapture::render(
                            *device,
                            {
                                .geometry = geometryPool->buffer().handle(),
                                .draws = draws.data(),
                                .drawCount = sceneDrawCount,
                                .descriptors = descriptorTable->set(),
                                .pipeline = bakePipeline.get(),
                                .format = gpu::kFormatR32Sfloat,
                                .faceSize = kPointShadowFaceSize,
                                .cameraSlotBase = gpu::FrameRenderer::kFramesInFlight,
                                // "No occluder" = far beyond any radius.
                                .clearColor = {1.0e6f, 0.0f, 0.0f, 0.0f},
                                .mipLevels = 1,
                            });
                        if (!cubeResult) {
                            log::warn("Point-shadow bake failed for light {}: {}", i,
                                      cubeResult.error().message);
                            continue;
                        }
                        pointShadowImages[i] = std::move(cubeResult).value();
                        // Binding 27 is not update-after-bind; the caller
                        // idles + invalidates around this lambda already.
                        descriptorTable->writePointShadowMap(i, pointShadowImages[i]->view());
                        pl.params[1] = 1.0f;
                        ++baked;
                    }
                }
                const auto bakeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::steady_clock::now() - bakeStart)
                                        .count();
                log::info("Point-light shadows baked in {} ms: {} cube(s), {}^2 x6 faces",
                          bakeMs, baked, kPointShadowFaceSize);
            }
        };
    }

    platform::TargetDesc desc{
        .style = platform::WindowStyle::Decorated,
        .size = {1280, 720},
        .title = "Renderer Viewer",
    };
    auto targetResult = backend->createTarget(desc);
    if (!targetResult) {
        log::error("Failed to create presentation target: {}", targetResult.error().message);
        return 1;
    }
    auto target = std::move(targetResult).value();

    auto surfaceResult = backend->createVulkanSurface(instance->handle(), *target);
    if (!surfaceResult) {
        log::error("Surface creation failed: {}", surfaceResult.error().message);
        return 1;
    }

    const auto extent = target->sizeInPixels();
    auto swapchainResult = gpu::Swapchain::create(*instance, *device,
                                                  {
                                                      .surface = surfaceResult.value(),
                                                      .width = extent.width,
                                                      .height = extent.height,
                                                      .transparent = false,
                                                      .vsync = vsync,
                                                  });
    if (!swapchainResult) {
        log::error("Swapchain creation failed: {}", swapchainResult.error().message);
        return 1;
    }
    auto swapchain = std::move(swapchainResult).value();

    // Shaders are compiled offline (dxc) into data/shaders next to the exe.
    const auto shaderDir = executableDirectory() / "data" / "shaders";

    // Scene pass pipeline: interleaved vertex input from the geometry pool,
    // depth-tested, camera via push constant.
    std::unique_ptr<gpu::Pipeline> transparentPipeline;
    std::unique_ptr<gpu::Pipeline> skyPipeline;
    std::unique_ptr<gpu::Pipeline> cullPipeline;
    std::unique_ptr<gpu::Pipeline> obbPipeline;
    std::unique_ptr<gpu::Pipeline> occlusionPipeline;
    std::unique_ptr<gpu::Pipeline> occlusionInstancePipeline;
    std::unique_ptr<gpu::Pipeline> occlusionDebugPipeline;
    std::unique_ptr<gpu::Pipeline> occlusionInstanceDebugPipeline;
    std::unique_ptr<gpu::Pipeline> shadowPipeline;
    std::unique_ptr<gpu::Pipeline> skinPipeline;
    std::unique_ptr<gpu::Pipeline> gbufferPipeline;
    std::unique_ptr<gpu::Pipeline> lightingPipeline;
    std::unique_ptr<gpu::Shader> sceneVert, sceneFrag, cullShader, obbShader, shadowVert,
        shadowFrag, skinShader, proxyVert, proxyInstVert, proxyFrag, proxyDebugFrag, skyVert,
        skyFrag,
        gbufferFrag, deferredVert, deferredFrag;
    if (scene) {
        auto vertResult = gpu::Shader::createFromFile(*device, shaderDir / "scene.vert.spv");
        // scene.hlsl's forward fragment shader survives ONLY for the
        // passes deferred cannot cover: the transparency blend pass here
        // and the probe/point-shadow captures (which load their own
        // modules). The RT variant traces shadow rays inline — loadable
        // only where the device enabled RayQuery.
        auto fragResult = gpu::Shader::createFromFile(
            *device, shaderDir / (rtReady ? "scene_rt.frag.spv" : "scene.frag.spv"));
        if (!vertResult || !fragResult) {
            log::error("{}", (!vertResult ? vertResult : fragResult).error().message);
            return 1;
        }
        sceneVert = std::move(vertResult).value();
        sceneFrag = std::move(fragResult).value();

        // Blend variant for the transparency pass: same shaders (the PS
        // outputs opacity for transparent-flagged fragments), alpha
        // blending on, depth write off.
        auto transparentResult = gpu::Pipeline::createGraphics(
            *device, {
                         .vertexShader = sceneVert.get(),
                         .fragmentShader = sceneFrag.get(),
                         .colorFormat = swapchain->imageFormat(),
                         .vertexStride = kVertexStride,
                         .vertexAttributes = {{0, gpu::kFormatR32G32B32Sfloat, 0},
                                              {1, gpu::kFormatR32G32B32Sfloat, 12},
                                              {2, gpu::kFormatR32G32Sfloat, 24}},
                         .depthFormat = gpu::kFormatD32Sfloat,
                         .pushConstantBytes = 2 * sizeof(std::uint32_t), // {slot, cascade}
                         .descriptorLayout = descriptorTable->layout(),
                         .alphaBlend = true,
                     });
        if (!transparentResult) {
            log::error("Transparent pipeline creation failed: {}",
                       transparentResult.error().message);
            return 1;
        }
        transparentPipeline = std::move(transparentResult).value();

        // Sky pass: fullscreen far-plane triangle painting the per-slot
        // light buffer's skyColor over background pixels — the dynamic
        // replacement for the baked clear color. Optional: failure just
        // leaves the static clear color as the background.
        auto skyVertResult = gpu::Shader::createFromFile(*device, shaderDir / "sky.vert.spv");
        auto skyFragResult = gpu::Shader::createFromFile(*device, shaderDir / "sky.frag.spv");
        if (skyVertResult && skyFragResult) {
            skyVert = std::move(skyVertResult).value();
            skyFrag = std::move(skyFragResult).value();
            auto skyResult = gpu::Pipeline::createGraphics(
                *device, {
                             .vertexShader = skyVert.get(),
                             .fragmentShader = skyFrag.get(),
                             .colorFormat = swapchain->imageFormat(),
                             .depthFormat = gpu::kFormatD32Sfloat,
                             .pushConstantBytes = 2 * sizeof(std::uint32_t), // {slot, cascade}
                             .descriptorLayout = descriptorTable->layout(),
                             .background = true,
                         });
            if (skyResult) {
                skyPipeline = std::move(skyResult).value();
            } else {
                log::warn("Sky pipeline unavailable: {}", skyResult.error().message);
            }
        } else {
            log::warn("Sky shaders unavailable: {}",
                      (!skyVertResult ? skyVertResult : skyFragResult).error().message);
        }

        // THE scene path: the G-buffer pass reuses scene.vert with the
        // attribute-MRT fragment shader; the lighting pass is a fullscreen
        // triangle Loading bindings 28-31. Fatal on failure — there is no
        // forward opaque fallback.
        auto gbufferFragResult =
            gpu::Shader::createFromFile(*device, shaderDir / "gbuffer.frag.spv");
        auto deferredVertResult =
            gpu::Shader::createFromFile(*device, shaderDir / "deferred.vert.spv");
        auto deferredFragResult = gpu::Shader::createFromFile(
            *device, shaderDir / (rtReady ? "deferred_rt.frag.spv" : "deferred.frag.spv"));
        if (!gbufferFragResult || !deferredVertResult || !deferredFragResult) {
            log::error("Deferred shaders missing: {}",
                       (!gbufferFragResult   ? gbufferFragResult
                        : !deferredVertResult ? deferredVertResult
                                              : deferredFragResult)
                           .error()
                           .message);
            return 1;
        }
        gbufferFrag = std::move(gbufferFragResult).value();
        deferredVert = std::move(deferredVertResult).value();
        deferredFrag = std::move(deferredFragResult).value();
        std::vector<std::uint32_t> gbufferFormats(gpu::FrameRenderer::kGBufferFormats.begin(),
                                                  gpu::FrameRenderer::kGBufferFormats.end());
        auto gbufferResult = gpu::Pipeline::createGraphics(
            *device, {
                         .vertexShader = sceneVert.get(),
                         .fragmentShader = gbufferFrag.get(),
                         .colorFormats = std::move(gbufferFormats),
                         .vertexStride = kVertexStride,
                         .vertexAttributes = {{0, gpu::kFormatR32G32B32Sfloat, 0},
                                              {1, gpu::kFormatR32G32B32Sfloat, 12},
                                              {2, gpu::kFormatR32G32Sfloat, 24}},
                         .depthFormat = gpu::kFormatD32Sfloat,
                         .pushConstantBytes = 2 * sizeof(std::uint32_t), // {slot, cascade}
                         .descriptorLayout = descriptorTable->layout(),
                     });
        auto lightingResult = gpu::Pipeline::createGraphics(
            *device, {
                         .vertexShader = deferredVert.get(),
                         .fragmentShader = deferredFrag.get(),
                         .colorFormat = swapchain->imageFormat(),
                         .depthFormat = gpu::kFormatD32Sfloat,
                         .pushConstantBytes = 2 * sizeof(std::uint32_t), // {slot, cascade}
                         .descriptorLayout = descriptorTable->layout(),
                         .disableDepthTest = true,
                     });
        if (!gbufferResult || !lightingResult) {
            log::error("Deferred pipeline creation failed: {}",
                       (!gbufferResult ? gbufferResult : lightingResult).error().message);
            return 1;
        }
        gbufferPipeline = std::move(gbufferResult).value();
        lightingPipeline = std::move(lightingResult).value();

        // The compaction pass (IndirectCount mode only). A failure here is
        // not fatal: the draw-mode ladder just skips to Indirect.
        auto cullShaderResult = gpu::Shader::createFromFile(*device, shaderDir / "cull.comp.spv");
        if (cullShaderResult) {
            cullShader = std::move(cullShaderResult).value();
            auto cullResult = gpu::Pipeline::createCompute(
                *device, {
                             .shader = cullShader.get(),
                             .descriptorLayout = descriptorTable->layout(),
                             // {drawCount, slot, capacity, flags, lodFactor}
                             .pushConstantBytes = 5 * sizeof(std::uint32_t),
                         });
            if (cullResult) {
                cullPipeline = std::move(cullResult).value();
            } else {
                log::warn("Cull pipeline unavailable: {}", cullResult.error().message);
            }
        } else {
            log::warn("Cull shader unavailable: {}", cullShaderResult.error().message);
        }

        // Incremental OBB refinement (IndirectCount mode): a baked
        // self-terminating dispatch tightening a few entries' boxes per
        // frame, fully on the GPU. Optional like the cull pipeline —
        // failure (or --noobb) just keeps the plain AABBs forever.
        if (obbRefine && cullPipeline) {
            auto obbShaderResult =
                gpu::Shader::createFromFile(*device, shaderDir / "obb.comp.spv");
            if (obbShaderResult) {
                obbShader = std::move(obbShaderResult).value();
                auto obbResult2 = gpu::Pipeline::createCompute(
                    *device, {
                                 .shader = obbShader.get(),
                                 .descriptorLayout = descriptorTable->layout(),
                                 // {drawCount, slot, capacity}
                                 .pushConstantBytes = 3 * sizeof(std::uint32_t),
                             });
                if (obbResult2) {
                    obbPipeline = std::move(obbResult2).value();
                } else {
                    log::warn("OBB refine pipeline unavailable: {}",
                              obbResult2.error().message);
                }
            } else {
                log::warn("OBB refine shader unavailable: {}",
                          obbShaderResult.error().message);
            }
        }

        // Occlusion proxy pipeline (IndirectCount mode only): AABB cubes
        // generated in the vertex shader, depth TEST only against the
        // frame's finished scene depth, color writes masked — exists for
        // the fragment shader's visibility stores alone. Optional like
        // the cull pipeline: failure just leaves occlusion off.
        auto proxyVertResult = gpu::Shader::createFromFile(*device, shaderDir / "proxy.vert.spv");
        auto proxyFragResult = gpu::Shader::createFromFile(*device, shaderDir / "proxy.frag.spv");
        if (!device->isEnabled(gpu::Feature::FragmentStores)) {
            log::warn("Occlusion culling unavailable: fragmentStoresAndAtomics not enabled");
        } else if (proxyVertResult && proxyFragResult) {
            proxyVert = std::move(proxyVertResult).value();
            proxyFrag = std::move(proxyFragResult).value();
            auto proxyResult = gpu::Pipeline::createGraphics(
                *device, {
                             .vertexShader = proxyVert.get(),
                             .fragmentShader = proxyFrag.get(),
                             .colorFormat = swapchain->imageFormat(),
                             .depthFormat = gpu::kFormatD32Sfloat,
                             .pushConstantBytes = 2 * sizeof(std::uint32_t), // {slot, capacity}
                             .descriptorLayout = descriptorTable->layout(),
                             .occlusionProxy = true,
                         });
            if (proxyResult) {
                occlusionPipeline = std::move(proxyResult).value();
                // Per-instance variant (runtime instanced models): same
                // PS, VS resolves canonical rows through binding 34 and
                // transforms local boxes. Optional like everything here.
                auto instVertResult = gpu::Shader::createFromFile(
                    *device, shaderDir / "proxy_instances.vert.spv");
                if (instVertResult) {
                    proxyInstVert = std::move(instVertResult).value();
                    auto instResult = gpu::Pipeline::createGraphics(
                        *device, {
                                     .vertexShader = proxyInstVert.get(),
                                     .fragmentShader = proxyFrag.get(),
                                     .colorFormat = swapchain->imageFormat(),
                                     .depthFormat = gpu::kFormatD32Sfloat,
                                     .pushConstantBytes =
                                         2 * sizeof(std::uint32_t), // {slot, capacity}
                                     .descriptorLayout = descriptorTable->layout(),
                                     .occlusionProxy = true,
                                 });
                    if (instResult) {
                        occlusionInstancePipeline = std::move(instResult).value();
                    } else {
                        log::warn("Occlusion instance pipeline unavailable: {}",
                                  instResult.error().message);
                    }
                } else {
                    log::warn("Occlusion instance shader unavailable: {}",
                              instVertResult.error().message);
                }
                // Debug-overlay variant: same VS/boxes, translucent color
                // instead of visibility stores (Settings -> "Show
                // occlusion boxes"). Optional like everything here.
                auto debugFragResult =
                    gpu::Shader::createFromFile(*device, shaderDir / "proxy_debug.frag.spv");
                if (debugFragResult) {
                    proxyDebugFrag = std::move(debugFragResult).value();
                    auto debugResult = gpu::Pipeline::createGraphics(
                        *device, {
                                     .vertexShader = proxyVert.get(),
                                     .fragmentShader = proxyDebugFrag.get(),
                                     .colorFormat = swapchain->imageFormat(),
                                     .depthFormat = gpu::kFormatD32Sfloat,
                                     .pushConstantBytes =
                                         2 * sizeof(std::uint32_t), // {slot, capacity}
                                     .descriptorLayout = descriptorTable->layout(),
                                     .occlusionDebug = true,
                                 });
                    if (debugResult) {
                        occlusionDebugPipeline = std::move(debugResult).value();
                    } else {
                        log::warn("Occlusion debug pipeline unavailable: {}",
                                  debugResult.error().message);
                    }
                    if (proxyInstVert) {
                        auto instDebugResult = gpu::Pipeline::createGraphics(
                            *device,
                            {
                                .vertexShader = proxyInstVert.get(),
                                .fragmentShader = proxyDebugFrag.get(),
                                .colorFormat = swapchain->imageFormat(),
                                .depthFormat = gpu::kFormatD32Sfloat,
                                .pushConstantBytes = 2 * sizeof(std::uint32_t),
                                .descriptorLayout = descriptorTable->layout(),
                                .occlusionDebug = true,
                            });
                        if (instDebugResult) {
                            occlusionInstanceDebugPipeline = std::move(instDebugResult).value();
                        } else {
                            log::warn("Occlusion instance debug pipeline unavailable: {}",
                                      instDebugResult.error().message);
                        }
                    }
                } else {
                    log::warn("Occlusion debug shader unavailable: {}",
                              debugFragResult.error().message);
                }
            } else {
                log::warn("Occlusion proxy pipeline unavailable: {}",
                          proxyResult.error().message);
            }
        } else {
            log::warn("Occlusion proxy shaders unavailable: {}",
                      (!proxyVertResult ? proxyVertResult : proxyFragResult).error().message);
        }

        // GPU skinning pass (animated scenes only): failure falls back to
        // the frozen rest pose baked at load.
        if (!animatedMeshes.empty()) {
            auto skinShaderResult =
                gpu::Shader::createFromFile(*device, shaderDir / "skin.comp.spv");
            if (skinShaderResult) {
                skinShader = std::move(skinShaderResult).value();
                auto skinResult = gpu::Pipeline::createCompute(
                    *device,
                    {
                        .shader = skinShader.get(),
                        .descriptorLayout = descriptorTable->layout(),
                        .pushConstantBytes =
                            gpu::DrawBatch::kSkinPushWords * sizeof(std::uint32_t),
                    });
                if (skinResult) {
                    skinPipeline = std::move(skinResult).value();
                } else {
                    log::warn("Skin pipeline unavailable: {}", skinResult.error().message);
                }
            } else {
                log::warn("Skin shader unavailable: {}", skinShaderResult.error().message);
            }
        }

        // Depth-only pipeline for the shadow pass: same vertex layout and
        // bindless set, no color attachment.
        auto shadowVertResult = gpu::Shader::createFromFile(*device, shaderDir / "shadow.vert.spv");
        auto shadowFragResult = gpu::Shader::createFromFile(*device, shaderDir / "shadow.frag.spv");
        if (shadowVertResult && shadowFragResult) {
            shadowVert = std::move(shadowVertResult).value();
            shadowFrag = std::move(shadowFragResult).value();
            auto shadowPipeResult = gpu::Pipeline::createGraphics(
                *device, {
                             .vertexShader = shadowVert.get(),
                             .fragmentShader = shadowFrag.get(),
                             .colorFormat = 0, // depth-only
                             .vertexStride = kVertexStride,
                             .vertexAttributes = {{0, gpu::kFormatR32G32B32Sfloat, 0},
                                                  {1, gpu::kFormatR32G32B32Sfloat, 12},
                                                  {2, gpu::kFormatR32G32Sfloat, 24}},
                             .depthFormat = gpu::kFormatD32Sfloat,
                             .pushConstantBytes = 2 * sizeof(std::uint32_t), // {slot, cascade}
                             .descriptorLayout = descriptorTable->layout(),
                         });
            if (shadowPipeResult) {
                shadowPipeline = std::move(shadowPipeResult).value();
            } else {
                log::warn("Shadow pipeline unavailable: {}", shadowPipeResult.error().message);
            }
        } else {
            log::warn("Shadow shaders unavailable: {}",
                      (!shadowVertResult ? shadowVertResult : shadowFragResult).error().message);
        }
    }

    // Traced primary visibility: fullscreen pass whose fragments walk the
    // TLAS (ps_6_5 — only loadable where RayQuery is enabled). Optional:
    // failure just leaves the raster path.
    std::unique_ptr<gpu::Pipeline> rtPrimaryPipeline;
    std::unique_ptr<gpu::Shader> rtPrimaryVert, rtPrimaryFrag;
    if (scene && rtReady) {
        auto vertResult = gpu::Shader::createFromFile(*device, shaderDir / "rt_primary.vert.spv");
        auto fragResult = gpu::Shader::createFromFile(*device, shaderDir / "rt_primary.frag.spv");
        if (vertResult && fragResult) {
            rtPrimaryVert = std::move(vertResult).value();
            rtPrimaryFrag = std::move(fragResult).value();
            auto pipeResult = gpu::Pipeline::createGraphics(
                *device, {
                             .vertexShader = rtPrimaryVert.get(),
                             .fragmentShader = rtPrimaryFrag.get(),
                             .colorFormat = swapchain->imageFormat(),
                             .depthFormat = 0, // rays need no depth buffer
                             .pushConstantBytes = 2 * sizeof(std::uint32_t),
                             .descriptorLayout = descriptorTable->layout(),
                         });
            if (pipeResult) {
                rtPrimaryPipeline = std::move(pipeResult).value();
            } else {
                log::warn("RT primary pipeline unavailable: {}", pipeResult.error().message);
            }
        } else {
            log::warn("RT primary shaders unavailable: {}",
                      (!vertResult ? vertResult : fragResult).error().message);
        }
    }

    auto rendererResult = gpu::FrameRenderer::create(*device, *swapchain);
    if (!rendererResult) {
        log::error("Frame renderer creation failed: {}", rendererResult.error().message);
        return 1;
    }
    auto renderer = std::move(rendererResult).value();

    // Deferred G-buffer targets (bindings 28-31), kept at swapchain extent
    // by the resize path. Non-fatal: deferred shading just stays
    // unavailable.
    if (descriptorTable) {
        if (auto r = renderer->setDeferredTargets(descriptorTable.get()); !r) {
            log::warn("Deferred targets unavailable: {}", r.error().message);
        }
    }

    // With a scene, every frame is the indirect batch over the geometry
    // pool; without one, frames present only the clear color + UI.
    const bool drawScene = gbufferPipeline && lightingPipeline && indirectBuffer &&
                           !geometry.empty();
    gpu::DrawBatch batch;
    if (drawScene) {
        // Pick the best submit mode the device's enabled features allow,
        // never exceeding the --draw-mode cap. Each tier falls back to the
        // next; Direct works everywhere.
        const bool canIndirect = device->isEnabled(gpu::Feature::MultiDrawIndirect) &&
                                 device->isEnabled(gpu::Feature::DrawIndirectFirstInstance);
        // The count tier now IS the compaction pass, so it also needs the
        // cull pipeline to have built.
        const bool canCount = canIndirect && device->isEnabled(gpu::Feature::DrawIndirectCount) &&
                              cullPipeline != nullptr;
        batch.mode = gpu::DrawSubmitMode::Direct;
        if (canIndirect && maxDrawMode != gpu::DrawSubmitMode::Direct) {
            batch.mode = gpu::DrawSubmitMode::Indirect;
        }
        if (canCount && maxDrawMode == gpu::DrawSubmitMode::IndirectCount) {
            batch.mode = gpu::DrawSubmitMode::IndirectCount;
        }

        batch.geometry = geometryPool->buffer().handle();
        batch.drawCount = static_cast<std::uint32_t>(geometry.size());
        batch.indirectRegionStride = templateCapacity * sizeof(gpu::DrawIndexedIndirect);
        if (batch.mode == gpu::DrawSubmitMode::IndirectCount) {
            // The GPU draws what the cull pass compacted, not the
            // templates: shadows the visibility-only list, the scene pass
            // the frustum-culled one.
            batch.indirect = compactedBuffer->handle();
            batch.sceneIndirect = culledBuffer->handle();
            batch.transparentIndirect = transparentBuffer->handle();
            batch.transparentPipeline = transparentPipeline.get();
            batch.cullPipeline = cullPipeline.get();
            batch.obbRefinePipeline = obbPipeline.get();
            batch.occlusionPipeline = occlusionPipeline.get();
            batch.occlusionInstancePipeline = occlusionInstancePipeline.get();
            batch.occlusionInstanceRows = kInstanceRowCapacity;
            batch.occlusionDebugPipeline =
                showOcclusionBoxes ? occlusionDebugPipeline.get() : nullptr;
            batch.occlusionInstanceDebugPipeline =
                showOcclusionBoxes ? occlusionInstanceDebugPipeline.get() : nullptr;
            batch.occlusionVisibility = visibilityBuffer->handle();
            // Region = entry slots + per-instance slots (matches the
            // hardcoded split in cull.hlsl / proxy.hlsl).
            batch.occlusionRegionStride =
                (std::uint64_t{templateCapacity} + kInstanceRowCapacity) *
                sizeof(std::uint32_t);
            batch.cullFlags = (frustumCull ? 1u : 0u) | (lodSelect ? 2u : 0u) |
                              (occlusionCull && occlusionPipeline ? 4u : 0u);
        } else {
            batch.indirect = indirectBuffer->handle();
        }
        // The only scene path: the G-buffer pass rasterizes whatever
        // stream the tier provides (culled opaque / templates / CPU
        // draws), the lighting triangle shades it.
        batch.gbufferPipeline = gbufferPipeline.get();
        batch.lightingPipeline = lightingPipeline.get();
        batch.count = countBuffer->handle();
        batch.countRegionStride = 8 * sizeof(std::uint32_t);
        batch.cpuDraws = draws.data();
        batch.descriptors = descriptorTable->set();
        batch.skyPipeline = skyPipeline.get();
        if (shadowPipeline && !shadowMaps.empty() &&
            SunControls::sceneSun(scene->lights).castsShadows) {
            batch.shadowPipeline = shadowPipeline.get();
            for (std::uint32_t c = 0; c < kShadowCascades; ++c) {
                batch.shadowCascades[c] = shadowMaps[c].get();
            }
            batch.cascadeCount = kShadowCascades;
        }
        batch.rtPrimaryPipeline = rtPrimaryPipeline.get();
        batch.rtPrimary = rtPrimaryFromStart && rtPrimaryPipeline != nullptr;
        if (skinPipeline && !animatedMeshes.empty()) {
            if (batch.mode == gpu::DrawSubmitMode::Direct) {
                log::warn("Animation needs per-slot indirect entries; Direct mode shows the "
                          "rest pose only");
            } else {
                batch.skinPipeline = skinPipeline.get();
                for (const AnimatedMeshEntry& entry : animatedMeshes) {
                    std::uint32_t jointBase = 0;
                    for (const AnimatedModelState& state : animatedStates) {
                        if (state.modelIndex == entry.modelIndex) {
                            jointBase = state.jointBase;
                            break;
                        }
                    }
                    gpu::DrawBatch::SkinDispatch dispatch;
                    dispatch.push = {entry.srcVertex,
                                     entry.dstVertexBase,
                                     entry.vertexCount,
                                     entry.skinVertexOffset,
                                     jointBase,
                                     entry.morphBase,
                                     entry.morphTargetCount,
                                     entry.morphWeightOffset,
                                     0, // slot, patched at record time
                                     totalJoints,
                                     static_cast<std::uint32_t>(morphWeightsFrame.size())};
                    dispatch.vertexCount = entry.vertexCount;
                    batch.skinDispatches.push_back(dispatch);
                }
                // With a BVH present, refit it from the freshly posed
                // vertices every frame so traced shadows/primary follow
                // the animation instead of the bind pose.
                if (rtReady && !refitGeometries.empty()) {
                    batch.refitBlas = blas.get();
                    batch.refitGeometries = refitGeometries;
                }
            }
        }
        if (rtReady) {
            // Rebuild the dynamic TLAS in place each frame from the slot's
            // instance region: runtime model spawns/moves/unloads reach
            // every traced path without touching recordings (and animated
            // scenes' refitted BLAS AABBs are re-read for free).
            batch.tlasRebuild = tlas.get();
        }

        const char* modeName = batch.mode == gpu::DrawSubmitMode::IndirectCount
                                   ? "indirect-count + GPU compaction"
                               : batch.mode == gpu::DrawSubmitMode::Indirect ? "indirect"
                                                                             : "direct";
        log::info("Scene pass ready: {} draws, {} mode (device: indirect {}, indirect-count {})",
                  batch.drawCount, modeName, canIndirect ? "yes" : "NO",
                  canCount ? "yes" : "NO");
    }

    // Debug UI (ImGui): drawn through the frame renderer's overlay pass,
    // which stays per-frame even when the scene buffers are static.
    auto ui = viewer::Ui::create(*instance, *device, *swapchain);
    if (ui) {
        renderer->setOverlayRecorder([&ui](VkCommandBuffer cmd) { ui->render(cmd); });
    } else {
        log::warn("Debug UI unavailable; continuing without it");
    }

    renderer->setStaticRecording(staticMode);
    log::info("Viewer live at {}x{} — {} recording, vsync {} — Esc quits, Space toggles mode",
              extent.width, extent.height, staticMode ? "static" : "per-frame", vsync ? "on" : "off");

    // Cull-pass counters read back from the frame slot's last completed
    // frame (IndirectCount mode): [0] the shadow passes' visibility-only
    // stream, [1] the scene pass's frustum-culled stream, [2] scratch
    // rows written for partially visible instanced draws, [3] transparent
    // draws, [4]/[5] indices emitted to the opaque/transparent streams
    // (post-LOD, /3 = triangles — the LOD A/B stat), [6] entries dropped
    // by the occlusion test. Stats only — read after the slot's fence,
    // kFramesInFlight frames late.
    std::uint32_t lastDrawCounts[7] = {0, 0, 0, 0, 0, 0, 0};
    // Settings-panel copies of the counters above, refreshed 4x per
    // second (matching the debug panel's readout cadence) — per-frame
    // stat text flickers unreadably. Graphs and bench reports keep the
    // live values.
    std::uint32_t shownDrawCounts[7] = {0, 0, 0, 0, 0, 0, 0};
    float statsRefreshTimer = 1.0f; // expired: first frame fills

    // Stats window: wall time + renderer CPU counters, reported per mode.
    constexpr std::uint64_t kReportInterval = 600;
    auto reportStart = std::chrono::steady_clock::now();
    auto report = [&](std::uint64_t windowFrames) {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - reportStart).count();
        reportStart = now;
        const gpu::FrameRenderer::Stats stats = renderer->takeStats();
        if (windowFrames == 0 || seconds <= 0.0) {
            return;
        }
        log::info("[{}] {} frames in {:.2f} s = {:.0f} fps | record {:.1f} us/frame{}{}",
                  renderer->staticRecording() ? "static" : "rerecord", windowFrames, seconds,
                  windowFrames / seconds,
                  stats.frames > 0 ? static_cast<double>(stats.recordMicros) / stats.frames : 0.0,
                  stats.prerecords > 0 ? std::format(" | {} prerecords", stats.prerecords) : "",
                  batch.cullPipeline
                      ? std::format(" | draws {}+{}t/{} in view, {} partial rows, {:.2f}M tris"
                                    ", {} occluded",
                                    lastDrawCounts[1], lastDrawCounts[3], lastDrawCounts[0],
                                    lastDrawCounts[2],
                                    (lastDrawCounts[4] + lastDrawCounts[5]) / 3.0e6,
                                    lastDrawCounts[6])
                      : "");
    };

    FlyCamera camera;
    SunControls sun;
    // Script-controllable sky/ambient/point-light state, written into the
    // per-slot light buffer every frame (defaults = the old constants).
    std::array<float, 3> skyColor{0.02f, 0.02f, 0.04f};
    std::array<float, 3> ambientColor{0.30f, 0.32f, 0.36f};
    // Skybox day phase in [0,1) shading the sky pass's procedural cube;
    // negative (the default) keeps the flat skyColor background.
    float timeOfDay = -1.0f;
    // Global fade over every point light's authored intensity
    // (set_point_light_scale — the script day/night lamp control).
    float pointLightScale = 1.0f;
    // Reflection technique for reflective-tagged objects in the raster
    // path; defaults to the best offer (traced where available, so nothing
    // visually regresses vs. the per-object RT milestone).
    bool reflectionsTraced = false;
    // Fly-in countdown (scene camera's flySeconds); 0 = no fly-in or done.
    float flyRemaining = 0.0f;
    if (scene) {
        camera = FlyCamera::fromScene(scene->camera);
        flyRemaining = scene->camera.flySeconds;
        if (flyRemaining > 0.0f) {
            camera.position = {scene->camera.flyFrom[0], scene->camera.flyFrom[1],
                               scene->camera.flyFrom[2]};
        }
        sun = SunControls::fromLight(SunControls::sceneSun(scene->lights));
        sun.rtShadows = rtFromStart && rtReady;
        // Scene-XML point lights fill the dynamic slots in declaration
        // order; scripts fade them via set_point_light_scale (or replace
        // slots wholesale via set_point_light).
        std::uint32_t pointSlot = 0;
        for (const assetio::LightDesc& light : scene->lights) {
            if (light.type != assetio::LightType::Point) {
                continue;
            }
            if (pointSlot >= kMaxPointLights) {
                log::warn("Scene declares more than {} point lights; extras dropped",
                          kMaxPointLights);
                break;
            }
            PointLight& slot = pointLights[pointSlot++];
            slot.positionRadius = {light.position[0], light.position[1], light.position[2],
                                   light.radius};
            slot.colorIntensity = {light.color[0], light.color[1], light.color[2],
                                   light.intensity};
            slot.params = {light.castsShadows ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f};
        }
        if (pointSlot > 0) {
            log::info("Scene point lights: {} ({} shadow-casting)", pointSlot,
                      std::count_if(pointLights.begin(), pointLights.end(),
                                    [](const PointLight& p) { return p.params[0] > 0.0f; }));
        }
        reflectionsTraced = rtReady && !forceProbeReflections;
        log::info("Sun: azimuth {:.0f}, elevation {:.0f}, intensity {:.2f}, shadows {}",
                  sun.azimuthDeg, sun.elevationDeg, sun.intensity,
                  batch.shadowPipeline ? "on" : "off");
        if (scene->fog.enabled) {
            // The raster background is a clear color, never a shaded pixel,
            // so match it to the fog's converged in-scatter (isotropic
            // phase + the shader's ambient term) — sky pixels then read as
            // "fog all the way out" instead of punching a clear hole.
            const assetio::FogDesc& fog = scene->fog;
            const float sunScatter = sun.intensity / (4.0f * 3.14159265f);
            const std::array<float, 3> ambient{0.15f, 0.16f, 0.18f};
            for (int i = 0; i < 3; ++i) {
                skyColor[i] = fog.color[i] * (sun.color[i] * sunScatter + ambient[i]);
            }
            renderer->setClearColor(skyColor[0], skyColor[1], skyColor[2]);
            log::info("Volumetric fog: box ({:.0f} {:.0f} {:.0f}) size ({:.0f} {:.0f} {:.0f}), "
                      "density {:.3f}, anisotropy {:.2f}, {} steps",
                      fog.position[0], fog.position[1], fog.position[2], fog.size[0],
                      fog.size[1], fog.size[2], fog.density, fog.anisotropy,
                      static_cast<int>(fog.steps));
        }
    }
    // ---- Runtime model loading over the renderer message queue ----
    // The viewer is the first producer (Settings panel + --spawn-test);
    // Python and other clients speak the same Command/Event schema later.
    // Commands drain once per frame at the safe point (after the frame
    // slot's fence), where the current slot's buffers are CPU-writable.
    renderer::MessageQueue messageQueue;
    renderer::Sender messageSender = messageQueue.createSender();
    // Events broadcast to per-consumer receivers; the Python host holds
    // one. The viewer itself currently consumes none (a receiver nothing
    // polls would only accumulate, so don't create one idly).
    // Resources vs instances (the dedup design in ARCHITECTURE.md): one
    // GeometryResource per asset path owns the pool slices, object rows,
    // and indirect entries; each LoadModel handle is an INSTANCE carrying
    // only a transform row. Repeat loads of a resident path skip import,
    // decode, and pool allocation entirely — they bump instanceCount and
    // write one instance row per mesh. The resource's GPU state is freed
    // when the last instance unloads (refcount to zero, deferred).
    // Identity is the exact path string, same as the texture dedup.
    struct ResourceInstance {
        renderer::ModelHandle handle = renderer::kInvalidModel;
        std::uint32_t transformIndex = 0;
    };
    struct QueuedInstance {
        renderer::LoadModelCmd cmd{};
        std::chrono::steady_clock::time_point requested;
    };
    struct GeometryResource {
        bool resident = false;
        // Reflective tag requested by the FIRST load of the path — object
        // rows are per resource, so every instance shares it; later loads
        // asking for the other value warn and get this one.
        std::uint32_t reflective = 0;
        std::vector<std::uint32_t> meshObjectIndices; // rows in draws/objectData
        // Local-space AABB per mesh: scene-AABB growth per placed instance.
        std::vector<std::pair<math::Vec3, math::Vec3>> meshBounds;
        // Instance rows: one block of meshCount * instanceCapacity rows;
        // mesh m's draw has firstInstance = instanceBase + m * capacity.
        std::uint32_t instanceBase = 0;
        std::uint32_t instanceCapacity = 0;
        std::vector<ResourceInstance> instances;
        std::vector<QueuedInstance> queued; // arrivals while still loading
        // RT state (rtReady only): a local-space BLAS over the meshes plus
        // this resource's block in the hit->object remap table. remapCount
        // 0 = raster-only (build failed / remap exhausted) — instances
        // simply never enter the TLAS.
        std::unique_ptr<gpu::AccelerationStructure> rtBlas;
        std::uint32_t remapBase = 0;
        std::uint32_t remapCount = 0;
    };
    std::unordered_map<std::string, GeometryResource> resourcesByPath;
    // Live instance handles in spawn order (the test harness unloads the
    // oldest and moves random picks).
    std::vector<renderer::ModelHandle> liveHandles;
    // Per-frame dynamic-TLAS instance list (entry 0 = the scene BLAS),
    // rebuilt from resourcesByPath and written into the current slot's
    // instance region each frame.
    std::vector<gpu::AccelerationStructure::Instance> rtInstances;
    bool rtInstancesOverflowWarned = false;
    // Slots whose template region must be resynced from the canonical
    // `draws` (instance add/remove edits instanceCounts; each slot syncs
    // when it is the current one — its region is not in flight then).
    std::array<bool, gpu::FrameRenderer::kFramesInFlight> templatesDirty{};
    // Deferred destruction for unloads: pool slices, draw-table rows,
    // transform rows, and instance-row blocks return to circulation
    // kFramesInFlight drains after release, when the last submission that
    // could still read them has retired. Loads recycle retired rows before
    // appending, so steady-state churn holds every table flat.
    struct PendingReclaim {
        std::vector<gpu::BufferSlice> slices;
        std::vector<std::uint32_t> objectIndices;
        std::vector<std::uint32_t> transformIndices;
        std::uint32_t instanceRowBase = 0;
        std::uint32_t instanceRowCount = 0; // 0 = no block to free
        // Retiring RT state: the BLAS destructs when the reclaim pops
        // (in-flight TLAS instance regions may still reference it until
        // then), the remap block returns to its free list.
        std::unique_ptr<gpu::AccelerationStructure> rtBlas;
        std::uint32_t remapBase = 0;
        std::uint32_t remapCount = 0;
        std::uint64_t retireAtDrain = 0;
    };
    std::deque<PendingReclaim> pendingReclaims;
    std::vector<std::uint32_t> freeObjectIndices;
    std::vector<std::uint32_t> freeTransformIndices;
    std::uint64_t drainFrame = 0;

    // Transform rows: recycle freed indices, grow past the scene prefix
    // otherwise. The canonical vector's size is the high-water mark the
    // per-frame slot memcpy covers.
    auto allocateTransformIndex = [&]() -> std::optional<std::uint32_t> {
        if (!freeTransformIndices.empty()) {
            const std::uint32_t index = freeTransformIndices.back();
            freeTransformIndices.pop_back();
            return index;
        }
        if (objectTransforms.size() >= kTransformCapacity) {
            return std::nullopt;
        }
        objectTransforms.push_back(kIdentityMat4);
        return static_cast<std::uint32_t>(objectTransforms.size() - 1);
    };

    // Instance-row blocks: first-fit over a sorted, coalesced free list,
    // growing past the scene prefix when no hole fits.
    struct RowRange {
        std::uint32_t base = 0;
        std::uint32_t count = 0;
    };
    std::vector<RowRange> freeInstanceRanges;
    std::uint32_t instanceRowHighWater = static_cast<std::uint32_t>(geometry.size());
    auto allocateInstanceRows = [&](std::uint32_t count) -> std::optional<std::uint32_t> {
        for (auto it = freeInstanceRanges.begin(); it != freeInstanceRanges.end(); ++it) {
            if (it->count >= count) {
                const std::uint32_t base = it->base;
                it->base += count;
                it->count -= count;
                if (it->count == 0) {
                    freeInstanceRanges.erase(it);
                }
                return base;
            }
        }
        if (instanceRowHighWater + count <= kInstanceRowCapacity) {
            const std::uint32_t base = instanceRowHighWater;
            instanceRowHighWater += count;
            return base;
        }
        return std::nullopt;
    };
    // Mark canonical rows dead in the row->entry map (binding 34): the
    // per-instance proxy pass skips them. 0xffffffff = dead.
    auto markRowEntriesDead = [&](std::uint32_t base, std::uint32_t count) {
        auto* mapped = static_cast<std::uint32_t*>(rowEntryBuffer->mapped());
        for (std::uint32_t i = 0; i < count; ++i) {
            mapped[base + i] = 0xffffffffu;
        }
    };
    auto freeInstanceRows = [&](std::uint32_t base, std::uint32_t count) {
        if (count == 0) {
            return;
        }
        // Cleared at RETIREMENT, not at relocation/unload — in-flight
        // templates may draw the old rows for kFramesInFlight more
        // frames, and the per-instance proxy must keep marking their
        // visibility bits or those instances would occlusion-drop for a
        // frame (visible flicker on every capacity growth).
        markRowEntriesDead(base, count);
        auto it = std::lower_bound(
            freeInstanceRanges.begin(), freeInstanceRanges.end(), base,
            [](const RowRange& range, std::uint32_t b) { return range.base < b; });
        it = freeInstanceRanges.insert(it, {base, count});
        if (auto next = std::next(it);
            next != freeInstanceRanges.end() && it->base + it->count == next->base) {
            it->count += next->count;
            freeInstanceRanges.erase(next);
        }
        if (it != freeInstanceRanges.begin()) {
            auto prev = std::prev(it);
            if (prev->base + prev->count == it->base) {
                prev->count += it->count;
                freeInstanceRanges.erase(it);
            }
        }
        if (!freeInstanceRanges.empty()) {
            const RowRange& last = freeInstanceRanges.back();
            if (last.base + last.count == instanceRowHighWater) {
                instanceRowHighWater = last.base;
                freeInstanceRanges.pop_back();
            }
        }
    };
    // RT remap blocks (binding 32): same first-fit/coalescing shape as the
    // instance rows, over kRuntimeObjectCapacity entries. Blocks must be
    // contiguous — a TLAS instance's customIndex + GeometryIndex lands
    // inside its resource's block.
    std::vector<RowRange> freeRemapRanges;
    std::uint32_t remapHighWater = 0;
    auto allocateRemapBlock = [&](std::uint32_t count) -> std::optional<std::uint32_t> {
        for (auto it = freeRemapRanges.begin(); it != freeRemapRanges.end(); ++it) {
            if (it->count >= count) {
                const std::uint32_t base = it->base;
                it->base += count;
                it->count -= count;
                if (it->count == 0) {
                    freeRemapRanges.erase(it);
                }
                return base;
            }
        }
        if (remapHighWater + count <= kRuntimeObjectCapacity) {
            const std::uint32_t base = remapHighWater;
            remapHighWater += count;
            return base;
        }
        return std::nullopt;
    };
    auto freeRemapBlock = [&](std::uint32_t base, std::uint32_t count) {
        if (count == 0) {
            return;
        }
        auto it = std::lower_bound(
            freeRemapRanges.begin(), freeRemapRanges.end(), base,
            [](const RowRange& range, std::uint32_t b) { return range.base < b; });
        it = freeRemapRanges.insert(it, {base, count});
        if (auto next = std::next(it);
            next != freeRemapRanges.end() && it->base + it->count == next->base) {
            it->count += next->count;
            freeRemapRanges.erase(next);
        }
        if (it != freeRemapRanges.begin()) {
            auto prev = std::prev(it);
            if (prev->base + prev->count == it->base) {
                prev->count += it->count;
                freeRemapRanges.erase(it);
            }
        }
        if (!freeRemapRanges.empty()) {
            const RowRange& last = freeRemapRanges.back();
            if (last.base + last.count == remapHighWater) {
                remapHighWater = last.base;
                freeRemapRanges.pop_back();
            }
        }
    };
    auto writeInstanceRow = [&](std::uint32_t index, InstanceRow row) {
        instanceRows[index] = row;
        std::memcpy(static_cast<std::byte*>(instanceRowBuffer->mapped()) +
                        index * sizeof(InstanceRow),
                    &row, sizeof(InstanceRow));
        // Runtime rows' objectIndex IS the draw entry — keep the
        // row->entry map (per-instance proxy pass) in lockstep.
        static_cast<std::uint32_t*>(rowEntryBuffer->mapped())[index] = row.objectIndex;
    };
    auto placementMatrix = [&](const float position[3], float yawDegrees, float scale) {
        const float yaw = yawDegrees * kPi / 180.0f;
        return composeTrs({position[0], position[1], position[2]},
                          {0.0f, std::sin(yaw * 0.5f), 0.0f, std::cos(yaw * 0.5f)},
                          {scale, scale, scale});
    };
    // Grow the scene AABB (shadow cascade fitting) by a mesh's local
    // bounds pushed through a placement matrix.
    auto growSceneAabb = [&](const math::Vec3& localMin, const math::Vec3& localMax,
                             const math::Mat4& m) {
        for (int corner = 0; corner < 8; ++corner) {
            const float x = (corner & 1) ? localMax.x : localMin.x;
            const float y = (corner & 2) ? localMax.y : localMin.y;
            const float z = (corner & 4) ? localMax.z : localMin.z;
            const math::Vec3 w{m[0] * x + m[4] * y + m[8] * z + m[12],
                               m[1] * x + m[5] * y + m[9] * z + m[13],
                               m[2] * x + m[6] * y + m[10] * z + m[14]};
            sceneMin = {std::min(sceneMin.x, w.x), std::min(sceneMin.y, w.y),
                        std::min(sceneMin.z, w.z)};
            sceneMax = {std::max(sceneMax.x, w.x), std::max(sceneMax.y, w.y),
                        std::max(sceneMax.z, w.z)};
        }
    };
    // Cascade-fit AABB = static scene bounds union every LIVE runtime
    // instance. Spawn/move/unload mark it dirty; the frame loop
    // recomputes before fitting, so a moved instance is followed and an
    // unloaded one stops inflating the fit (previously the union only
    // grew on spawn and SetTransform never fed it at all).
    bool sceneAabbDirty = false;
    auto recomputeSceneAabb = [&] {
        sceneMin = staticSceneMin;
        sceneMax = staticSceneMax;
        for (const auto& [path, resource] : resourcesByPath) {
            for (const ResourceInstance& instance : resource.instances) {
                const math::Mat4& m = objectTransforms[instance.transformIndex];
                for (const auto& [localMin, localMax] : resource.meshBounds) {
                    growSceneAabb(localMin, localMax, m);
                }
            }
        }
    };
    // Runtime resources carry their LOCAL per-mesh AABBs in the bounds
    // table (mode bmax[3] = 1): the cull shader pushes them through each
    // instance's transform itself, so nothing here tracks instance
    // motion — SetTransform only ever touches the transform row.

    // Rewrites one slot's whole template region from the canonical draw
    // list, reapplying that slot's animated posed-vertex overrides.
    auto writeTemplates = [&](std::uint32_t slot) {
        std::vector<gpu::DrawIndexedIndirect> slotDraws = draws;
        for (const AnimatedMeshEntry& entry : animatedMeshes) {
            slotDraws[entry.objectIndex].vertexOffset =
                static_cast<std::int32_t>(entry.dstVertexBase + slot * entry.vertexCount);
        }
        std::memcpy(static_cast<std::byte*>(indirectBuffer->mapped()) +
                        slot * batch.indirectRegionStride,
                    slotDraws.data(), slotDraws.size() * sizeof(gpu::DrawIndexedIndirect));
    };

    // Adds one instance of a RESIDENT resource: a transform row, one
    // instance row per mesh, and instanceCount bumps. This is the whole
    // cost of a repeat LoadModel — no import, no decode, no pool touch.
    auto addInstance = [&](GeometryResource& resource, const renderer::LoadModelCmd& cmd,
                           std::uint32_t slot) -> Result<void> {
        const auto meshCount = static_cast<std::uint32_t>(resource.meshObjectIndices.size());
        const auto count = static_cast<std::uint32_t>(resource.instances.size());
        if (count == resource.instanceCapacity) {
            // Grow by relocating the per-mesh row blocks. In-flight slots'
            // templates keep the old firstInstance; the old block's rows
            // stay untouched until their deferred reclaim retires.
            const std::uint32_t newCapacity = std::max(4u, resource.instanceCapacity * 2u);
            const auto newBase = allocateInstanceRows(meshCount * newCapacity);
            if (!newBase) {
                return Error{std::format("instance rows exhausted ({} of {} in use)",
                                         instanceRowHighWater, kInstanceRowCapacity)};
            }
            for (std::uint32_t m = 0; m < meshCount; ++m) {
                for (std::uint32_t k = 0; k < count; ++k) {
                    writeInstanceRow(*newBase + m * newCapacity + k,
                                     {.objectIndex = resource.meshObjectIndices[m],
                                      .transformIndex = resource.instances[k].transformIndex});
                }
                draws[resource.meshObjectIndices[m]].firstInstance =
                    *newBase + m * newCapacity;
            }
            templatesDirty.fill(true); // canonical firstInstance changed
            if (resource.instanceCapacity > 0) {
                PendingReclaim reclaim;
                reclaim.instanceRowBase = resource.instanceBase;
                reclaim.instanceRowCount = meshCount * resource.instanceCapacity;
                reclaim.retireAtDrain = drainFrame + gpu::FrameRenderer::kFramesInFlight;
                pendingReclaims.push_back(std::move(reclaim));
            }
            resource.instanceBase = *newBase;
            resource.instanceCapacity = newCapacity;
        }
        const auto transformIndex = allocateTransformIndex();
        if (!transformIndex) {
            return Error{
                std::format("transform capacity exhausted ({} rows)", kTransformCapacity)};
        }
        const math::Mat4 placement = placementMatrix(cmd.position, cmd.yawDegrees, cmd.scale);
        objectTransforms[*transformIndex] = placement;
        for (std::uint32_t m = 0; m < meshCount; ++m) {
            // The new row sits past every in-flight template's
            // instanceCount, so writing it now is safe.
            writeInstanceRow(resource.instanceBase + m * resource.instanceCapacity + count,
                             {.objectIndex = resource.meshObjectIndices[m],
                              .transformIndex = *transformIndex});
            draws[resource.meshObjectIndices[m]].instanceCount = count + 1;
        }
        sceneAabbDirty = true; // the cascade fit picks up the new instance
        resource.instances.push_back({.handle = cmd.handle, .transformIndex = *transformIndex});
        liveHandles.push_back(cmd.handle);
        templatesDirty.fill(true);
        writeTemplates(slot);
        templatesDirty[slot] = false;
        if (batch.mode == gpu::DrawSubmitMode::Direct) {
            renderer->invalidateStaticRecordings(); // Direct mode bakes cpuDraws
        }
        return {};
    };

    // Removes instance k of a resource; when it was the last one, the
    // resource's whole GPU state rides the same deferred reclaim. The
    // caller erases the emptied resource from the map.
    auto removeInstanceAt = [&](GeometryResource& resource, std::size_t k) {
        sceneAabbDirty = true; // the cascade fit stops covering it
        const auto meshCount = static_cast<std::uint32_t>(resource.meshObjectIndices.size());
        const std::size_t last = resource.instances.size() - 1;
        PendingReclaim reclaim;
        reclaim.transformIndices.push_back(resource.instances[k].transformIndex);
        if (k != last) {
            // Swap-remove. An in-flight slot still drawing last+1 rows sees
            // the moved instance twice for a frame (same transform, benign)
            // and loses the removed one a frame early.
            for (std::uint32_t m = 0; m < meshCount; ++m) {
                writeInstanceRow(resource.instanceBase + m * resource.instanceCapacity +
                                     static_cast<std::uint32_t>(k),
                                 {.objectIndex = resource.meshObjectIndices[m],
                                  .transformIndex = resource.instances[last].transformIndex});
            }
            resource.instances[k] = resource.instances[last];
        }
        resource.instances.pop_back();
        for (std::uint32_t m = 0; m < meshCount; ++m) {
            draws[resource.meshObjectIndices[m]].instanceCount =
                static_cast<std::uint32_t>(resource.instances.size());
            // The vacated tail row's transformIndex will eventually be
            // recycled by another model — stop the per-instance proxy
            // from drawing a ghost box on it. In-flight templates still
            // drawing the old count lose the (duplicate) tail copy's
            // visibility bit a frame early — benign, same window as the
            // swap-remove duplicate itself.
            markRowEntriesDead(resource.instanceBase + m * resource.instanceCapacity +
                                   static_cast<std::uint32_t>(last),
                               1);
        }
        templatesDirty.fill(true);
        if (batch.mode == gpu::DrawSubmitMode::Direct) {
            renderer->invalidateStaticRecordings();
        }
        if (resource.instances.empty()) {
            for (std::uint32_t index : resource.meshObjectIndices) {
                reclaim.slices.push_back(geometry[index].vertices);
                reclaim.slices.push_back(geometry[index].indices);
            }
            reclaim.objectIndices = std::move(resource.meshObjectIndices);
            reclaim.instanceRowBase = resource.instanceBase;
            reclaim.instanceRowCount = meshCount * resource.instanceCapacity;
            reclaim.rtBlas = std::move(resource.rtBlas);
            reclaim.remapBase = resource.remapBase;
            reclaim.remapCount = resource.remapCount;
        }
        reclaim.retireAtDrain = drainFrame + gpu::FrameRenderer::kFramesInFlight;
        pendingReclaims.push_back(std::move(reclaim));
    };

    // ---- Async loader (milestone 2) ----
    // The expensive parts of a runtime load — glTF import, CPU interleave,
    // texture decode (~2 s cold, ~50 ms warm) — run on a dedicated worker
    // thread; the frame loop never touches files. The drain only INTEGRATES
    // finished loads: pool alloc + staging + upload flush + template append
    // (~2 ms, GPU objects are main-thread-only). ModelReady fires at
    // integration, so its millis is request→integrated wall time.
    struct PreparedTexture {
        std::string path;
        bool srgb = true;
        bool decoded = false;
        assetio::TextureData data;
    };
    struct PreparedMesh {
        std::vector<float> vertexData; // interleaved, ready for the pool
        std::vector<std::uint32_t> indices;
        std::uint32_t materialIndex = 0;
        ObjectData object; // texture slots resolved at integration by path
        std::string baseColorPath;
        std::string normalPath;
        std::string mrPath;
        math::Vec3 localMin{1e30f, 1e30f, 1e30f};
        math::Vec3 localMax{-1e30f, -1e30f, -1e30f};
    };
    struct PreparedLoad {
        renderer::LoadModelCmd cmd{};
        std::chrono::steady_clock::time_point requested;
        std::string error; // non-empty: preparation failed on the worker
        std::vector<PreparedMesh> meshes;
        std::vector<PreparedTexture> textures; // freshly decoded, uncached
    };
    struct PendingLoad {
        renderer::LoadModelCmd cmd{};
        std::chrono::steady_clock::time_point requested;
    };
    std::mutex loaderMutex; // guards loadRequests + loaderQuit
    std::condition_variable loaderWake;
    std::deque<PendingLoad> loadRequests;
    bool loaderQuit = false;
    std::mutex preparedMutex;
    std::vector<PreparedLoad> preparedLoads;
    // Texture paths decoded (or being decoded) by ANYONE — seeded with the
    // scene's uploads. The worker claims paths here before decoding;
    // textureSlotByPath itself stays main-thread-only. A claim without a
    // map entry yet means "an earlier in-flight load carries the bytes",
    // which FIFO integration resolves before anything can look it up.
    std::mutex textureClaimMutex;
    std::unordered_set<std::string> claimedTexturePaths;
    for (const auto& [path, slot] : textureSlotByPath) {
        claimedTexturePaths.insert(path);
    }

    // Worker-side preparation: everything that needs no GPU objects.
    auto prepareLoad = [&](PendingLoad&& pending) {
        REND_PROFILE_ZONE("PrepareLoad");
        PreparedLoad out{.cmd = pending.cmd, .requested = pending.requested};
        auto imported = [&] {
            REND_PROFILE_ZONE("LoaderImport");
            return importers.import(std::filesystem::path(out.cmd.path));
        }();
        if (!imported) {
            out.error = imported.error().message;
            return out;
        }
        assetio::ModelData data = std::move(imported).value();
        bool animated = !data.skeleton.empty() || !data.animations.empty();
        for (const auto& mesh : data.meshes) {
            animated = animated || mesh.skinned || !mesh.morphTargets.empty();
        }
        if (animated) {
            out.error = "animated models are not supported at runtime yet";
            return out;
        }
        auto claimTexture = [&](const std::filesystem::path& path, bool srgb) {
            if (path.empty()) {
                return;
            }
            std::lock_guard lock(textureClaimMutex);
            if (claimedTexturePaths.insert(path.string()).second) {
                out.textures.push_back({.path = path.string(), .srgb = srgb});
            }
        };
        for (auto& mesh : data.meshes) {
            PreparedMesh prepared;
            prepared.materialIndex = mesh.materialIndex;
            if (mesh.materialIndex < data.materials.size()) {
                const auto& material = data.materials[mesh.materialIndex];
                prepared.object.flags = (material.alphaMasked ? kObjectAlphaMasked : 0u) |
                                        (material.transparent ? kObjectTransparent : 0u);
                prepared.object.alphaCutoff = material.alphaCutoff;
                prepared.object.baseAlpha = material.baseColorFactor[3];
                prepared.object.metallicFactor = material.metallicFactor;
                prepared.object.roughnessFactor = material.roughnessFactor;
                prepared.baseColorPath = material.baseColorTexture.string();
                prepared.normalPath = material.normalTexture.string();
                prepared.mrPath = material.metallicRoughnessTexture.string();
                claimTexture(material.baseColorTexture, true);
                claimTexture(material.normalTexture, false);
                claimTexture(material.metallicRoughnessTexture, false);
            }
            // Semantic scene tag, same as the XML path's reflective="true":
            // the RT-variant shaders decide whether to trace it.
            if (out.cmd.reflective != 0) {
                prepared.object.flags |= kObjectReflective;
            }
            prepared.vertexData = interleave(mesh);
            for (std::size_t v = 0; v + 2 < mesh.positions.size(); v += 3) {
                prepared.localMin.x = std::min(prepared.localMin.x, mesh.positions[v]);
                prepared.localMin.y = std::min(prepared.localMin.y, mesh.positions[v + 1]);
                prepared.localMin.z = std::min(prepared.localMin.z, mesh.positions[v + 2]);
                prepared.localMax.x = std::max(prepared.localMax.x, mesh.positions[v]);
                prepared.localMax.y = std::max(prepared.localMax.y, mesh.positions[v + 1]);
                prepared.localMax.z = std::max(prepared.localMax.z, mesh.positions[v + 2]);
            }
            prepared.indices = std::move(mesh.indices);
            out.meshes.push_back(std::move(prepared));
        }
        // Decode claimed textures in parallel (same atomic-counter pool as
        // the scene load); failures release the claim so a later spawn
        // retries instead of resolving to white forever.
        if (!out.textures.empty()) {
            REND_PROFILE_ZONE("LoaderTextureDecode");
            std::atomic<std::size_t> nextTexture{0};
            const std::size_t workerCount = std::min<std::size_t>(
                out.textures.size(), std::max(1u, std::thread::hardware_concurrency()));
            std::vector<std::thread> decodePool;
            decodePool.reserve(workerCount);
            for (std::size_t w = 0; w < workerCount; ++w) {
                decodePool.emplace_back([&] {
                    for (std::size_t i = nextTexture.fetch_add(1); i < out.textures.size();
                         i = nextTexture.fetch_add(1)) {
                        PreparedTexture& texture = out.textures[i];
                        if (auto result = assetio::loadTexture(texture.path)) {
                            texture.data = std::move(result).value();
                            texture.decoded = true;
                        }
                    }
                });
            }
            for (std::thread& worker : decodePool) {
                worker.join();
            }
            std::erase_if(out.textures, [&](const PreparedTexture& texture) {
                if (texture.decoded) {
                    return false;
                }
                log::warn("Runtime texture '{}' failed to decode", texture.path);
                std::lock_guard lock(textureClaimMutex);
                claimedTexturePaths.erase(texture.path);
                return true;
            });
        }
        return out;
    };

    std::thread loaderThread([&] {
        REND_PROFILE_THREAD("loader");
        for (;;) {
            PendingLoad pending;
            {
                std::unique_lock lock(loaderMutex);
                loaderWake.wait(lock, [&] { return loaderQuit || !loadRequests.empty(); });
                if (loaderQuit) {
                    return; // pending requests are dropped on shutdown
                }
                pending = std::move(loadRequests.front());
                loadRequests.pop_front();
            }
            PreparedLoad prepared = prepareLoad(std::move(pending));
            std::lock_guard lock(preparedMutex);
            preparedLoads.push_back(std::move(prepared));
        }
    });

    // ModelReady for one instance handle: event + the log line whose pool
    // and draw-table figures the churn tests watch.
    auto pushModelReady = [&](renderer::ModelHandle handle,
                              std::chrono::steady_clock::time_point requested,
                              const Result<void>& applied, const char* path,
                              std::size_t instanceCount) {
        renderer::Event event;
        event.type = renderer::Event::Type::ModelReady;
        event.ready.handle = handle;
        event.ready.ok = applied.ok();
        event.ready.millis =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() -
                                                     requested)
                .count();
        if (applied) {
            log::info("Runtime model '{}' ready in {:.1f} ms (handle {}) | pool {:.1f} MiB "
                      "in {} allocations, {} draw entries, {} instance(s) of this asset",
                      path, event.ready.millis, handle,
                      static_cast<double>(geometryPool->usedBytes()) / (1024.0 * 1024.0),
                      geometryPool->allocationCount(), draws.size(), instanceCount);
        } else {
            std::snprintf(event.ready.error, sizeof(event.ready.error), "%s",
                          applied.error().message.c_str());
            log::warn("Runtime model '{}' failed: {}", path, applied.error().message);
        }
        messageQueue.pushEvent(event);
    };

    // Main-thread integration of one prepared load: builds the RESOURCE
    // (textures, pool slices, object rows, indirect entries), then adds
    // every instance queued behind it. The only remaining frame-loop cost
    // of a cold runtime load.
    auto integrateResource = [&](PreparedLoad& load, std::uint32_t slot) {
        REND_PROFILE_ZONE("RuntimeLoadModel");
        const std::string path{load.cmd.path};
        auto resourceIt = resourcesByPath.find(path);
        auto failQueued = [&](const Result<void>& applied) {
            if (resourceIt == resourcesByPath.end()) {
                pushModelReady(load.cmd.handle, load.requested, applied, load.cmd.path, 0);
                return;
            }
            for (const QueuedInstance& queued : resourceIt->second.queued) {
                pushModelReady(queued.cmd.handle, queued.requested, applied, load.cmd.path, 0);
            }
            resourcesByPath.erase(resourceIt);
        };
        auto releaseClaims = [&] {
            std::lock_guard lock(textureClaimMutex);
            for (const PreparedTexture& texture : load.textures) {
                claimedTexturePaths.erase(texture.path);
            }
        };
        if (resourceIt == resourcesByPath.end()) {
            // Shouldn't happen (only integrate erases loading entries),
            // but never dereference end(): drop the load, free the claims.
            releaseClaims();
            log::warn("Prepared load '{}' has no resource entry; dropped", load.cmd.path);
            return;
        }
        if (!load.error.empty()) {
            releaseClaims();
            failQueued(Error{load.error});
            return;
        }
        // Only rows the free list can't cover actually grow the table.
        const std::size_t appended =
            load.meshes.size() - std::min(freeObjectIndices.size(), load.meshes.size());
        if (geometry.size() + appended > templateCapacity) {
            releaseClaims(); // carried decodes never upload; let retries re-decode
            failQueued(Error{std::format("runtime capacity exhausted ({} + {} > {})",
                                         geometry.size(), appended, templateCapacity)});
            return;
        }
        // Upload this load's carried textures first so its own meshes (and
        // any later load that saw the claim) can resolve them by path.
        // Binding 1 is UPDATE_AFTER_BIND + partially bound: writing a slot
        // no in-flight frame references is legal mid-run.
        for (PreparedTexture& texture : load.textures) {
            auto image = uploader
                             ? uploadTextureData(*uploader, texture.data, texture.srgb)
                             : Result<std::unique_ptr<gpu::Image>>{Error{"no uploader"}};
            if (!image) {
                log::warn("Runtime texture '{}' failed: {}", texture.path,
                          image.error().message);
                std::lock_guard lock(textureClaimMutex);
                claimedTexturePaths.erase(texture.path);
                continue;
            }
            const auto textureIndex = static_cast<std::uint32_t>(textures.size());
            descriptorTable->writeTexture(textureIndex, image.value()->view());
            textures.push_back(std::move(image).value());
            textureSlotByPath.emplace(texture.path, textureIndex);
        }
        auto textureSlot = [&](const std::string& texPath) -> std::uint32_t {
            if (texPath.empty()) {
                return 0u;
            }
            const auto it = textureSlotByPath.find(texPath);
            return it != textureSlotByPath.end() ? it->second : 0u;
        };

        const std::uint32_t oldDrawCount = static_cast<std::uint32_t>(draws.size());
        GeometryResource& resource = resourceIt->second;
        std::vector<gpu::BufferSlice> allocated; // freed straight back on failure
        auto fail = [&](Result<void> error) {
            // Nothing references the new state yet: slices go straight
            // back to the pool, rows straight to the free list (appended
            // rows stay in the table as recyclable instanceCount-0 rows,
            // template-initialized in every slot below).
            for (const gpu::BufferSlice& slice : allocated) {
                geometryPool->free(slice);
            }
            for (std::uint32_t index : resource.meshObjectIndices) {
                draws[index] = {};
                for (std::uint32_t s = 0; s < gpu::FrameRenderer::kFramesInFlight; ++s) {
                    if (index >= oldDrawCount) {
                        std::memcpy(static_cast<std::byte*>(indirectBuffer->mapped()) +
                                        s * batch.indirectRegionStride +
                                        index * sizeof(gpu::DrawIndexedIndirect),
                                    &draws[index], sizeof(gpu::DrawIndexedIndirect));
                    }
                }
                freeObjectIndices.push_back(index);
            }
            batch.drawCount = static_cast<std::uint32_t>(draws.size());
            batch.cpuDraws = draws.data();
            failQueued(error);
        };
        for (PreparedMesh& mesh : load.meshes) {
            // Placement rides the per-instance transform buffer; vertices
            // stay in model space and are shared by every instance.
            mesh.object.textureIndex = textureSlot(mesh.baseColorPath);
            mesh.object.normalIndex = textureSlot(mesh.normalPath);
            mesh.object.mrIndex = textureSlot(mesh.mrPath);

            auto vertexSlice =
                geometryPool->allocate(mesh.vertexData.size() * sizeof(float), kVertexStride);
            auto indexSlice =
                geometryPool->allocate(mesh.indices.size() * sizeof(std::uint32_t), 4);
            if (!vertexSlice || !indexSlice) {
                if (vertexSlice) {
                    allocated.push_back(vertexSlice.value());
                }
                fail(Error{std::format(
                    "pool allocation failed: {}",
                    (!vertexSlice ? vertexSlice.error() : indexSlice.error()).message)});
                return;
            }
            allocated.push_back(vertexSlice.value());
            allocated.push_back(indexSlice.value());
            auto stagedVerts =
                transfer->stage(geometryPool->buffer(), vertexSlice.value().offset,
                                mesh.vertexData.data(), vertexSlice.value().size);
            auto stagedIndices = transfer->stage(geometryPool->buffer(),
                                                 indexSlice.value().offset, mesh.indices.data(),
                                                 indexSlice.value().size);
            if (!stagedVerts || !stagedIndices) {
                fail(Error{std::format(
                    "staging failed: {}",
                    (!stagedVerts ? stagedVerts.error() : stagedIndices.error()).message)});
                return;
            }
            // Recycle a retired draw-table row when one is free; only
            // append when the free list is empty (bounds the table under
            // steady-state churn).
            std::uint32_t objectIndex;
            if (!freeObjectIndices.empty()) {
                objectIndex = freeObjectIndices.back();
                freeObjectIndices.pop_back();
            } else {
                objectIndex = static_cast<std::uint32_t>(geometry.size());
                geometry.emplace_back();
                objectData.emplace_back();
                draws.emplace_back();
                objectBounds.emplace_back();
            }
            geometry[objectIndex] = {.vertices = vertexSlice.value(),
                                     .indices = indexSlice.value(),
                                     .indexCount =
                                         static_cast<std::uint32_t>(mesh.indices.size()),
                                     .materialIndex = mesh.materialIndex};
            objectData[objectIndex] = mesh.object;
            draws[objectIndex] = {
                .indexCount = static_cast<std::uint32_t>(mesh.indices.size()),
                .instanceCount = 0, // instances bump this below
                .firstIndex = static_cast<std::uint32_t>(indexSlice.value().offset /
                                                         sizeof(std::uint32_t)),
                .vertexOffset =
                    static_cast<std::int32_t>(vertexSlice.value().offset / kVertexStride),
                .firstInstance = 0, // patched by the first addInstance's block alloc
            };
            resource.meshObjectIndices.push_back(objectIndex);
            resource.meshBounds.emplace_back(mesh.localMin, mesh.localMax);
            // Local bounds, per-instance mode: the cull shader transforms
            // and tests them against every instance's matrix each frame.
            // Transparent meshes route to the blend pass's stream.
            const float boundFlags =
                (mesh.object.flags & kObjectTransparent) != 0 ? kBoundsTransparent : 0.0f;
            objectBounds[objectIndex] = {
                .bmin = {mesh.localMin.x, mesh.localMin.y, mesh.localMin.z, boundFlags},
                .bmax = {mesh.localMax.x, mesh.localMax.y, mesh.localMax.z, 1.0f},
            };
        }
        // New SSBO rows (possibly scattered across recycled indices) ride
        // the same flush as the geometry.
        for (std::uint32_t index : resource.meshObjectIndices) {
            if (auto staged = transfer->stage(*objectBuffer, index * sizeof(ObjectData),
                                              &objectData[index], sizeof(ObjectData));
                !staged) {
                fail(staged.error());
                return;
            }
        }
        {
            REND_PROFILE_ZONE("RuntimeUploadFlush");
            if (auto flushed = transfer->flush(); !flushed) {
                fail(flushed.error());
                return;
            }
        }
        resource.resident = true;
        batch.drawCount = static_cast<std::uint32_t>(draws.size());
        batch.cpuDraws = draws.data();
        // RT: stage this resource's geometryInfo rows (hit-triangle fetch)
        // and its remap block (customIndex + GeometryIndex -> object row),
        // then build a local-space BLAS over the meshes — instances start
        // entering the dynamic TLAS on the next frame's instance write.
        // Any failure leaves the resource raster-only (warn, no event).
        if (rtReady) {
            const auto meshCount =
                static_cast<std::uint32_t>(resource.meshObjectIndices.size());
            const auto remapBase = allocateRemapBlock(meshCount);
            if (!remapBase) {
                log::warn("RT remap table exhausted; '{}' is raster-only in traced passes",
                          load.cmd.path);
            } else {
                std::vector<GeometryInfo> infos; // alive until the flush below
                infos.reserve(meshCount);
                bool rtOk = true;
                for (std::uint32_t index : resource.meshObjectIndices) {
                    infos.push_back({
                        .firstIndex = static_cast<std::uint32_t>(geometry[index].indices.offset /
                                                                 sizeof(std::uint32_t)),
                        .vertexOffset = static_cast<std::uint32_t>(
                            geometry[index].vertices.offset / kVertexStride),
                    });
                    if (auto staged = transfer->stage(*geometryInfoBuffer,
                                                      index * sizeof(GeometryInfo),
                                                      &infos.back(), sizeof(GeometryInfo));
                        !staged) {
                        rtOk = false;
                        break;
                    }
                }
                if (rtOk) {
                    if (auto staged = transfer->stage(
                            *rtRemapBuffer, *remapBase * sizeof(std::uint32_t),
                            resource.meshObjectIndices.data(),
                            meshCount * sizeof(std::uint32_t));
                        !staged) {
                        rtOk = false;
                    }
                }
                if (rtOk) {
                    if (auto flushed = transfer->flush(); !flushed) {
                        log::warn("RT info upload failed: {}", flushed.error().message);
                        rtOk = false;
                    }
                }
                if (rtOk) {
                    std::vector<gpu::AccelerationStructure::TriangleGeometry> triangles;
                    triangles.reserve(meshCount);
                    for (std::uint32_t index : resource.meshObjectIndices) {
                        triangles.push_back({
                            .buffer = &geometryPool->buffer(),
                            .vertexOffset = geometry[index].vertices.offset,
                            .vertexStride = kVertexStride,
                            .vertexCount = static_cast<std::uint32_t>(
                                geometry[index].vertices.size / kVertexStride),
                            .indexOffset = geometry[index].indices.offset,
                            .indexCount = geometry[index].indexCount,
                            .opaque = (objectData[index].flags &
                                       (kObjectAlphaMasked | kObjectTransparent)) == 0,
                        });
                    }
                    auto blasResult =
                        gpu::AccelerationStructure::buildBottomLevel(*device, triangles);
                    if (blasResult) {
                        resource.rtBlas = std::move(blasResult).value();
                        resource.remapBase = *remapBase;
                        resource.remapCount = meshCount;
                    } else {
                        log::warn("Runtime BLAS build failed: {}", blasResult.error().message);
                    }
                }
                if (!resource.rtBlas) {
                    freeRemapBlock(*remapBase, meshCount); // nothing referenced it
                }
            }
        }
        // Every instance queued behind the load lands now; each failure is
        // its own event, a success elsewhere in the queue still stands.
        std::vector<QueuedInstance> queued = std::move(resource.queued);
        resource.queued.clear();
        for (const QueuedInstance& entry : queued) {
            auto applied = addInstance(resource, entry.cmd, slot);
            pushModelReady(entry.cmd.handle, entry.requested, applied, load.cmd.path,
                           resource.instances.size());
        }
        // Template entries beyond the old drawCount are unread by any
        // in-flight frame: extend every slot region immediately (with the
        // final instanceCount/firstInstance the adds above produced).
        // Recycled entries sit BELOW it, where other slots' in-flight
        // frames still read — addInstance synced the current slot and
        // marked the others dirty; until their resync they read the
        // retired entry, which is instanceCount 0 (the model is simply
        // absent there one extra frame).
        for (std::uint32_t index : resource.meshObjectIndices) {
            if (index < oldDrawCount) {
                continue;
            }
            for (std::uint32_t s = 0; s < gpu::FrameRenderer::kFramesInFlight; ++s) {
                std::memcpy(static_cast<std::byte*>(indirectBuffer->mapped()) +
                                s * batch.indirectRegionStride +
                                index * sizeof(gpu::DrawIndexedIndirect),
                            &draws[index], sizeof(gpu::DrawIndexedIndirect));
            }
        }
        if (resource.instances.empty()) {
            // Every queued instance was unloaded or failed before the
            // resource finished loading: retire its GPU state right away.
            PendingReclaim reclaim;
            for (std::uint32_t index : resource.meshObjectIndices) {
                reclaim.slices.push_back(geometry[index].vertices);
                reclaim.slices.push_back(geometry[index].indices);
            }
            reclaim.objectIndices = std::move(resource.meshObjectIndices);
            reclaim.instanceRowBase = resource.instanceBase;
            reclaim.instanceRowCount =
                static_cast<std::uint32_t>(reclaim.objectIndices.size()) *
                resource.instanceCapacity;
            reclaim.rtBlas = std::move(resource.rtBlas);
            reclaim.remapBase = resource.remapBase;
            reclaim.remapCount = resource.remapCount;
            reclaim.retireAtDrain = drainFrame + gpu::FrameRenderer::kFramesInFlight;
            pendingReclaims.push_back(std::move(reclaim));
            resourcesByPath.erase(resourceIt);
        }
        // Static recordings bake the draw count; rebuild lazily (waits
        // idle — the ~2 ms integration hitch, paid once per NEW resource;
        // instance adds of resident resources never pay it).
        renderer->invalidateStaticRecordings();
    };

    auto applyUnloadModel = [&](const renderer::UnloadModelCmd& cmd) {
        for (auto it = resourcesByPath.begin(); it != resourcesByPath.end(); ++it) {
            GeometryResource& resource = it->second;
            if (!resource.resident) {
                // Queued behind a still-loading resource: drop it and tell
                // the producer, so a wait on the handle can't hang.
                for (auto queuedIt = resource.queued.begin();
                     queuedIt != resource.queued.end(); ++queuedIt) {
                    if (queuedIt->cmd.handle == cmd.handle) {
                        pushModelReady(cmd.handle, queuedIt->requested,
                                       Error{"unloaded before the resource finished loading"},
                                       it->first.c_str(), 0);
                        resource.queued.erase(queuedIt);
                        return;
                    }
                }
                continue;
            }
            for (std::size_t k = 0; k < resource.instances.size(); ++k) {
                if (resource.instances[k].handle != cmd.handle) {
                    continue;
                }
                removeInstanceAt(resource, k);
                std::erase(liveHandles, cmd.handle);
                if (resource.instances.empty()) {
                    resourcesByPath.erase(it);
                }
                return;
            }
        }
        log::warn("UnloadModel: unknown handle {}", cmd.handle);
    };

    auto processMessages = [&](std::uint32_t slot) {
        ++drainFrame;
        // Retire due reclaims first (we are past waitFrameSlot, so every
        // submission that could read these slices/rows has completed).
        while (!pendingReclaims.empty() &&
               drainFrame >= pendingReclaims.front().retireAtDrain) {
            PendingReclaim& reclaim = pendingReclaims.front();
            for (const gpu::BufferSlice& slice : reclaim.slices) {
                geometryPool->free(slice);
            }
            freeObjectIndices.insert(freeObjectIndices.end(), reclaim.objectIndices.begin(),
                                     reclaim.objectIndices.end());
            freeTransformIndices.insert(freeTransformIndices.end(),
                                        reclaim.transformIndices.begin(),
                                        reclaim.transformIndices.end());
            freeInstanceRows(reclaim.instanceRowBase, reclaim.instanceRowCount);
            // The retiring BLAS destructs with the reclaim below; its
            // remap block returns to circulation now.
            freeRemapBlock(reclaim.remapBase, reclaim.remapCount);
            log::trace("Reclaimed {} pool slices, {} instance rows; pool {:.1f} MiB in {} "
                       "allocations",
                       reclaim.slices.size(), reclaim.instanceRowCount,
                       static_cast<double>(geometryPool->usedBytes()) / (1024.0 * 1024.0),
                       geometryPool->allocationCount());
            pendingReclaims.pop_front();
        }
        if (templatesDirty[slot] && indirectBuffer) {
            writeTemplates(slot);
            templatesDirty[slot] = false;
        }
        for (const renderer::Command& cmd : messageQueue.drain()) {
            switch (cmd.type) {
            case renderer::Command::Type::LoadModel: {
                // Resource dedup: only the FIRST load of a path reaches the
                // loader thread. A repeat while it prepares queues behind
                // it; a repeat of a resident resource lands right now as a
                // pure instance add.
                const std::string path{cmd.load.path};
                const auto now = std::chrono::steady_clock::now();
                auto [it, inserted] = resourcesByPath.try_emplace(path);
                GeometryResource& resource = it->second;
                if (inserted) {
                    resource.reflective = cmd.load.reflective;
                } else if (resource.reflective != cmd.load.reflective) {
                    log::warn("load_model('{}'): reflective={} ignored — the resource is "
                              "shared and the first load chose {}",
                              path, cmd.load.reflective != 0, resource.reflective != 0);
                }
                if (inserted) {
                    resource.queued.push_back({cmd.load, now});
                    {
                        std::lock_guard lock(loaderMutex);
                        loadRequests.push_back({cmd.load, now});
                    }
                    loaderWake.notify_one();
                } else if (!resource.resident) {
                    resource.queued.push_back({cmd.load, now});
                } else {
                    auto applied = addInstance(resource, cmd.load, slot);
                    pushModelReady(cmd.load.handle, now, applied, cmd.load.path,
                                   resource.instances.size());
                }
                break;
            }
            case renderer::Command::Type::SetTransform: {
                const math::Mat4 placement =
                    placementMatrix(cmd.transform.position, cmd.transform.yawDegrees,
                                    cmd.transform.scale);
                bool found = false;
                for (auto& [path, resource] : resourcesByPath) {
                    for (ResourceInstance& instance : resource.instances) {
                        if (instance.handle == cmd.transform.handle) {
                            // The cull shader tests local bounds through
                            // this transform; the cascade fit re-unions.
                            objectTransforms[instance.transformIndex] = placement;
                            sceneAabbDirty = true;
                            found = true;
                            break;
                        }
                    }
                    if (found) {
                        break;
                    }
                }
                if (!found) {
                    log::warn("SetTransform: unknown handle {}", cmd.transform.handle);
                }
                break;
            }
            case renderer::Command::Type::UnloadModel:
                applyUnloadModel(cmd.unload);
                break;
            // Lighting commands replace viewer-side state the frame loop
            // copies into the per-slot light buffer every frame — no
            // recordings touched, no stalls (the sun's cascades refit on
            // the CPU each frame already).
            case renderer::Command::Type::SetSun: {
                assetio::LightDesc desc;
                desc.direction = {cmd.sun.direction[0], cmd.sun.direction[1],
                                  cmd.sun.direction[2]};
                desc.color = {cmd.sun.color[0], cmd.sun.color[1], cmd.sun.color[2]};
                desc.intensity = cmd.sun.intensity;
                const SunControls incoming = SunControls::fromLight(desc);
                sun.azimuthDeg = incoming.azimuthDeg;
                sun.elevationDeg = incoming.elevationDeg;
                sun.color = incoming.color;
                sun.intensity = incoming.intensity;
                break;
            }
            case renderer::Command::Type::SetSkyColor:
                skyColor = {cmd.sky.color[0], cmd.sky.color[1], cmd.sky.color[2]};
                break;
            case renderer::Command::Type::SetTimeOfDay:
                timeOfDay = cmd.timeOfDay.t;
                break;
            case renderer::Command::Type::SetAmbient:
                ambientColor = {cmd.ambient.color[0], cmd.ambient.color[1],
                                cmd.ambient.color[2]};
                break;
            case renderer::Command::Type::SetPointLight: {
                if (cmd.pointLight.index >= kMaxPointLights) {
                    log::warn("SetPointLight: index {} out of range (max {})",
                              cmd.pointLight.index, kMaxPointLights - 1);
                    break;
                }
                PointLight& light = pointLights[cmd.pointLight.index];
                light.positionRadius = {cmd.pointLight.position[0], cmd.pointLight.position[1],
                                        cmd.pointLight.position[2],
                                        std::max(cmd.pointLight.radius, 0.0f)};
                light.colorIntensity = {cmd.pointLight.color[0], cmd.pointLight.color[1],
                                        cmd.pointLight.color[2],
                                        std::max(cmd.pointLight.intensity, 0.0f)};
                // A replaced slot drops its baked cube (the light may have
                // moved); castsShadows keeps gating traced rays.
                light.params = {cmd.pointLight.castsShadows != 0 ? 1.0f : 0.0f, 0.0f, 0.0f,
                                0.0f};
                break;
            }
            case renderer::Command::Type::SetPointLightScale:
                pointLightScale = std::max(cmd.pointLightScale.scale, 0.0f);
                break;
            }
        }
        // Integrate whatever the loader finished, in completion order —
        // which the single worker keeps FIFO, so a load that skipped a
        // texture decode always integrates after the load carrying it.
        std::vector<PreparedLoad> ready;
        {
            std::lock_guard lock(preparedMutex);
            ready.swap(preparedLoads);
        }
        for (PreparedLoad& load : ready) {
            // Builds the resource, adds every queued instance, and fires
            // one ModelReady per queued handle (the log line's pool +
            // draw-table figures are the memory-churn test's metric).
            integrateResource(load, slot);
        }
    };

    // Test-harness producer state: nondeterministic auto-spawns driven by
    // the --spawn-test flag (the Python host is the interactive producer).
    char spawnPathBuf[512] = "C:/github/scenes/flight_helmet/model/FlightHelmet.gltf";
    if (spawnTestPath) {
        std::snprintf(spawnPathBuf, sizeof(spawnPathBuf), "%s", spawnTestPath);
    }
    bool autoSpawn = spawnTestPath != nullptr;
    float spawnIntervalMin = 0.5f, spawnIntervalMax = 2.5f;
    float nextSpawnIn = 0.5f;
    std::mt19937 spawnRng{1234}; // fixed seed: reproducible test runs
    auto requestSpawn = [&]() {
        const math::Vec3 f = camera.forward();
        const math::Vec3 right = math::normalize(math::cross(f, {0.0f, 1.0f, 0.0f}));
        std::uniform_real_distribution<float> side(-1.5f, 1.5f);
        std::uniform_real_distribution<float> angle(0.0f, 360.0f);
        const float s = side(spawnRng);
        renderer::Command cmd;
        cmd.type = renderer::Command::Type::LoadModel;
        cmd.load.handle = messageQueue.allocateHandle();
        std::snprintf(cmd.load.path, sizeof(cmd.load.path), "%s", spawnPathBuf);
        cmd.load.position[0] = camera.position.x + f.x * 3.0f + right.x * s;
        cmd.load.position[1] = camera.position.y - 0.5f;
        cmd.load.position[2] = camera.position.z + f.z * 3.0f + right.z * s;
        cmd.load.yawDegrees = angle(spawnRng);
        cmd.load.scale = 1.0f;
        messageSender.push(cmd);
        messageSender.flush();
    };

    // Scene-declared Python scripts: load the optional host DLL and run
    // them against the message queue on the host's own thread. A scene
    // without <Script> nodes never touches Python — the DLL isn't even
    // loaded, so it (and the CPython runtime) may be absent entirely.
    HMODULE pyhostDll = nullptr;
    pyhost::StopFn pyhostStop = nullptr;
    if (scene && !scene->scripts.empty()) {
        const std::filesystem::path dllPath = executableDirectory() / "rend_pyhost.dll";
        pyhostDll = LoadLibraryW(dllPath.wstring().c_str());
        const auto start = pyhostDll ? reinterpret_cast<pyhost::StartFn>(
                                           GetProcAddress(pyhostDll, "rendPyHostStart"))
                                     : nullptr;
        pyhostStop = pyhostDll ? reinterpret_cast<pyhost::StopFn>(
                                     GetProcAddress(pyhostDll, "rendPyHostStop"))
                               : nullptr;
        if (start && pyhostStop) {
            std::vector<std::string> scriptStorage;
            std::vector<const char*> scriptPtrs;
            for (const auto& path : scene->scripts) {
                scriptStorage.push_back(path.string());
            }
            for (const std::string& path : scriptStorage) {
                scriptPtrs.push_back(path.c_str());
            }
            if (start(&messageQueue, scriptPtrs.data(),
                      static_cast<int>(scriptPtrs.size()))) {
                log::info("Python host started ({} script(s))", scriptPtrs.size());
            } else {
                log::warn("Python host refused to start; scene scripts skipped");
                pyhostStop = nullptr;
            }
        } else {
            log::warn("Scene declares {} Python script(s) but rend_pyhost.dll is "
                      "unavailable; running without them",
                      scene->scripts.size());
            pyhostStop = nullptr;
        }
    }

    // Held-key state for camera movement; mouselook while RMB is held.
    bool keyHeld[static_cast<int>(platform::Key::LeftCtrl) + 1] = {};
    bool mouselook = false;
    constexpr float kLookSensitivity = 0.0025f; // radians per pixel
    constexpr float kMoveSpeed = 3.0f;          // units per second
    constexpr float kFastMultiplier = 5.0f;

    bool running = true;
    std::uint64_t frame = 0;
    std::uint32_t viewWidth = extent.width;
    std::uint32_t viewHeight = extent.height;
    auto lastFrameTime = std::chrono::steady_clock::now();
    while (running) {
        REND_PROFILE_ZONE("Frame");
        // Scene-texture streaming pump: a couple of uploads per frame off
        // the decode queue (graphics-queue one-shots, this thread), then
        // the accumulated bindless slot rewrites in batches under a
        // waitIdle sync point — the rewritten slots were sampled (as
        // white) by in-flight frames, so nothing may be pending during
        // the update. Rendering never blocks on the decode workers.
        if (texStream.active) {
            REND_PROFILE_ZONE("TextureStreamPump");
            constexpr std::size_t kUploadsPerFrame = 2;
            for (std::size_t n = 0; n < kUploadsPerFrame; ++n) {
                std::pair<std::size_t, Result<assetio::TextureData>> item{0, Error{""}};
                {
                    std::lock_guard lock(texStream.mutex);
                    if (texStream.ready.empty()) {
                        break;
                    }
                    item = std::move(texStream.ready.front());
                    texStream.ready.pop_front();
                }
                const std::size_t slot = item.first;
                ++texStream.uploaded;
                if (!item.second) {
                    log::warn("Texture {} failed ({}), slot stays white",
                              texturePaths[slot].path.filename().string(),
                              item.second.error().message);
                    continue;
                }
                texStream.texelBytes += item.second.value().bytes.size();
                auto image =
                    uploadTextureData(*uploader, item.second.value(), texturePaths[slot].srgb);
                if (!image) {
                    log::warn("Texture {} upload failed ({}), slot stays white",
                              texturePaths[slot].path.filename().string(),
                              image.error().message);
                    continue;
                }
                textures[slot] = std::move(image).value();
                texStream.pendingSlots.push_back(static_cast<std::uint32_t>(slot));
            }
            const bool complete = texStream.uploaded >= texStream.total;
            const auto now = std::chrono::steady_clock::now();
            if (!texStream.pendingSlots.empty() &&
                (complete || texStream.pendingSlots.size() >= 16 ||
                 now - texStream.lastFlush > std::chrono::milliseconds(400))) {
                renderer->waitIdle();
                for (std::uint32_t slot : texStream.pendingSlots) {
                    descriptorTable->writeTexture(slot, textures[slot]->view());
                }
                texStream.pendingSlots.clear();
                texStream.lastFlush = now;
            }
            if (complete) {
                texStream.active = false;
                for (std::thread& worker : texStream.workers) {
                    worker.join();
                }
                texStream.workers.clear();
                const auto texMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                       std::chrono::steady_clock::now() - texStream.start)
                                       .count();
                log::info("Textures ready in {} ms: {} images ({:.1f} MiB decoded), streamed, "
                          "bindless",
                          texMs, textures.size(),
                          static_cast<double>(texStream.texelBytes) / (1024.0 * 1024.0));
                if (waitForTextures) {
                    waitForTextures = false;
                    // The wait cover passed a null batch to drawFrame, so
                    // any static recording baked so far is the clear-only
                    // one — drop it here rather than relying on the probe
                    // branch's invalidate happening to run next frame.
                    renderer->invalidateStaticRecordings();
                }
            }
        } else if (capturePendingProbe) {
            // One-shot deferred probe capture, now that every texture the
            // capture samples is resident. Binding 18 is not update-after-
            // bind: idle around the write, re-record statics after.
            renderer->waitIdle();
            capturePendingProbe();
            capturePendingProbe = nullptr;
            renderer->invalidateStaticRecordings();
        }

        const auto events = [&] {
            REND_PROFILE_ZONE("PumpEvents");
            return backend->pumpEvents();
        }();
        for (const auto& event : events) {
            if (ui) {
                ui->handleEvent(event);
            }
            switch (event.type) {
            case platform::Event::Type::CloseRequested:
                running = false;
                break;
            case platform::Event::Type::KeyDown:
                keyHeld[static_cast<int>(event.key)] = true;
                if (event.key == platform::Key::Escape) {
                    running = false;
                }
                if (event.key == platform::Key::Space) {
                    report(frame % kReportInterval);
                    renderer->setStaticRecording(!renderer->staticRecording());
                    log::info("Switched to {} recording",
                              renderer->staticRecording() ? "static" : "per-frame");
                }
                break;
            case platform::Event::Type::KeyUp:
                keyHeld[static_cast<int>(event.key)] = false;
                break;
            case platform::Event::Type::MouseButtonDown:
                // The UI owns the mouse when the cursor is over a widget.
                if (event.button == platform::MouseButton::Right &&
                    !(ui && ImGui::GetIO().WantCaptureMouse)) {
                    mouselook = true;
                    backend->setRelativeMouseMode(*target, true);
                }
                break;
            case platform::Event::Type::MouseButtonUp:
                if (event.button == platform::MouseButton::Right) {
                    mouselook = false;
                    backend->setRelativeMouseMode(*target, false);
                }
                break;
            case platform::Event::Type::MouseMoved:
                if (mouselook) {
                    camera.yaw += event.mouseDeltaX * kLookSensitivity;
                    camera.pitch = std::clamp(camera.pitch - event.mouseDeltaY * kLookSensitivity,
                                              -0.49f * kPi, 0.49f * kPi);
                }
                break;
            case platform::Event::Type::Resized:
                renderer->resize(event.size.width, event.size.height);
                viewWidth = event.size.width;
                viewHeight = event.size.height;
                break;
            default:
                break;
            }
        }
        if (!running) {
            break;
        }

        const auto now = std::chrono::steady_clock::now();
        const float deltaSeconds = std::chrono::duration<float>(now - lastFrameTime).count();
        lastFrameTime = now;

        // Camera movement from held keys (fly style: WASD in the view
        // plane, Q/E down/up, Shift fast).
        if (drawScene) {
            auto held = [&](platform::Key k) { return keyHeld[static_cast<int>(k)]; };
            const float speed = kMoveSpeed *
                                (held(platform::Key::LeftShift) ? kFastMultiplier : 1.0f) *
                                deltaSeconds;
            const math::Vec3 f = camera.forward();
            const math::Vec3 right = math::normalize(math::cross(f, {0.0f, 1.0f, 0.0f}));
            auto move = [&](const math::Vec3& d, float s) {
                camera.position = {camera.position.x + d.x * s, camera.position.y + d.y * s,
                                   camera.position.z + d.z * s};
            };
            if (held(platform::Key::W)) move(f, speed);
            if (held(platform::Key::S)) move(f, -speed);
            if (held(platform::Key::D)) move(right, speed);
            if (held(platform::Key::A)) move(right, -speed);
            if (held(platform::Key::E)) move({0.0f, 1.0f, 0.0f}, speed);
            if (held(platform::Key::Q)) move({0.0f, 1.0f, 0.0f}, -speed);

            // Scene fly-in (<Camera flyFrom flySeconds>): ease from the
            // spawn point into the authored pose, aimed at the scene
            // target throughout; overrides input until it lands. Holds at
            // the spawn while a wait-mode loading cover hides the scene.
            if (scene && flyRemaining > 0.0f && !waitForTextures) {
                flyRemaining = std::max(flyRemaining - deltaSeconds, 0.0f);
                const float t = 1.0f - flyRemaining / scene->camera.flySeconds;
                const float e = t * t * (3.0f - 2.0f * t); // smoothstep
                const auto& from = scene->camera.flyFrom;
                const auto& to = scene->camera.position;
                camera.position = {from[0] + (to[0] - from[0]) * e,
                                   from[1] + (to[1] - from[1]) * e,
                                   from[2] + (to[2] - from[2]) * e};
                const math::Vec3 aim = math::normalize(math::sub(
                    {scene->camera.target[0], scene->camera.target[1], scene->camera.target[2]},
                    camera.position));
                camera.yaw = std::atan2(aim.x, -aim.z);
                camera.pitch = std::asin(std::clamp(aim.y, -1.0f, 1.0f));
            }

            // Publish this frame's camera into the slot's region: safe to
            // write once the slot's previous submission retired.
            if (auto r = renderer->waitFrameSlot(); !r) {
                log::error("Frame failed: {}", r.error().message);
                break;
            }
            // Streaming test: nondeterministic spawn requests, then the
            // frame's message drain (this slot's buffers are now safe).
            if (autoSpawn) {
                nextSpawnIn -= deltaSeconds;
                if (nextSpawnIn <= 0.0f) {
                    requestSpawn();
                    // Keep the population bounded: unload the oldest once
                    // more than four are live, so long test runs churn
                    // load AND unload instead of accumulating forever.
                    if (liveHandles.size() >= 4) {
                        renderer::Command cmd;
                        cmd.type = renderer::Command::Type::UnloadModel;
                        cmd.unload.handle = liveHandles.front();
                        messageSender.push(cmd);
                        messageSender.flush();
                    }
                    // Exercise SetTransform too: shove one live model to a
                    // fresh spot near the camera with a new heading.
                    if (!liveHandles.empty()) {
                        const auto pick = std::uniform_int_distribution<std::size_t>(
                            0, liveHandles.size() - 1)(spawnRng);
                        const math::Vec3 fwd = camera.forward();
                        const math::Vec3 side =
                            math::normalize(math::cross(fwd, {0.0f, 1.0f, 0.0f}));
                        const float s =
                            std::uniform_real_distribution<float>(-2.0f, 2.0f)(spawnRng);
                        renderer::Command cmd;
                        cmd.type = renderer::Command::Type::SetTransform;
                        cmd.transform.handle = liveHandles[pick];
                        cmd.transform.position[0] = camera.position.x + fwd.x * 4.0f + side.x * s;
                        cmd.transform.position[1] = camera.position.y - 0.5f;
                        cmd.transform.position[2] = camera.position.z + fwd.z * 4.0f + side.z * s;
                        cmd.transform.yawDegrees =
                            std::uniform_real_distribution<float>(0.0f, 360.0f)(spawnRng);
                        cmd.transform.scale = 1.0f;
                        messageSender.push(cmd);
                        messageSender.flush();
                    }
                    nextSpawnIn = std::uniform_real_distribution<float>(
                        spawnIntervalMin, spawnIntervalMax)(spawnRng);
                }
            }
            processMessages(renderer->frameSlot());
            // Publish this frame's TLAS instances into the slot's region:
            // the scene BLAS plus one entry per runtime model instance,
            // transforms straight from the canonical array — the recorded
            // per-frame rebuild makes spawns/moves/unloads reach every
            // traced path with no descriptor or recording churn.
            if (rtReady && tlas) {
                rtInstances.clear();
                rtInstances.push_back(
                    {.blas = blas.get(), .customIndex = kRtSceneCustomIndex});
                for (const auto& [resourcePath, resource] : resourcesByPath) {
                    if (!resource.rtBlas) {
                        continue;
                    }
                    for (const ResourceInstance& modelInstance : resource.instances) {
                        if (rtInstances.size() >= kRtInstanceCapacity) {
                            if (!rtInstancesOverflowWarned) {
                                log::warn("TLAS instance capacity ({}) exceeded; extra runtime "
                                          "instances are raster-only in traced passes",
                                          kRtInstanceCapacity);
                                rtInstancesOverflowWarned = true;
                            }
                            break;
                        }
                        rtInstances.push_back(
                            {.blas = resource.rtBlas.get(),
                             .customIndex = resource.remapBase,
                             .transform = objectTransforms[modelInstance.transformIndex]});
                    }
                }
                tlas->writeInstances(renderer->frameSlot(), rtInstances);
            }
            // Publish this frame's object transforms into the slot's region
            // (scene rows stay identity; runtime models carry placement).
            if (transformBuffer && !objectTransforms.empty()) {
                std::memcpy(static_cast<std::byte*>(transformBuffer->mapped()) +
                                std::uint64_t{renderer->frameSlot()} * kTransformCapacity *
                                    sizeof(math::Mat4),
                            objectTransforms.data(),
                            objectTransforms.size() * sizeof(math::Mat4));
            }
            // Cull-count stats: the slot's counts are from its last
            // completed frame (fenced) — this frame overwrites them later.
            if (countBuffer && batch.cullPipeline) {
                std::memcpy(lastDrawCounts,
                            static_cast<const std::byte*>(countBuffer->mapped()) +
                                renderer->frameSlot() * batch.countRegionStride,
                            sizeof(lastDrawCounts));
            }
            // Same for the cull-pass bounds: runtime instances move, so the
            // current slot's region tracks the canonical AABBs.
            if (boundsBuffer && !objectBounds.empty()) {
                std::memcpy(static_cast<std::byte*>(boundsBuffer->mapped()) +
                                std::uint64_t{renderer->frameSlot()} * templateCapacity *
                                    sizeof(ObjectBounds),
                            objectBounds.data(), objectBounds.size() * sizeof(ObjectBounds));
            }
            // Animation playback: sample every channel, rebuild node worlds
            // and joint matrices, and write this slot's regions (safe after
            // waitFrameSlot). The skinning pass consumes them this frame.
            if (batch.skinPipeline && !animatedStates.empty()) {
                REND_PROFILE_ZONE("Animation");
                for (AnimatedModelState& state : animatedStates) {
                    const auto& skeleton = state.data->skeleton;
                    const std::size_t nodeCount = skeleton.nodes.size();
                    state.t.resize(nodeCount);
                    state.r.resize(nodeCount);
                    state.s.resize(nodeCount);
                    state.world.resize(nodeCount);
                    for (std::size_t i = 0; i < nodeCount; ++i) {
                        state.t[i] = skeleton.nodes[i].translation;
                        state.r[i] = skeleton.nodes[i].rotation;
                        state.s[i] = skeleton.nodes[i].scale;
                    }
                    if (!state.data->animations.empty()) {
                        const auto& anim = state.data->animations.front();
                        if (anim.duration > 0.0f) {
                            state.time = std::fmod(state.time + deltaSeconds, anim.duration);
                        }
                        for (const auto& channel : anim.channels) {
                            switch (channel.path) {
                            case assetio::AnimationPath::Translation:
                                sampleChannel(channel, state.time,
                                              state.t[channel.node].data(), 3);
                                break;
                            case assetio::AnimationPath::Rotation:
                                sampleChannel(channel, state.time,
                                              state.r[channel.node].data(), 4);
                                break;
                            case assetio::AnimationPath::Scale:
                                sampleChannel(channel, state.time,
                                              state.s[channel.node].data(), 3);
                                break;
                            case assetio::AnimationPath::Weights:
                                for (const AnimatedMeshEntry& entry : animatedMeshes) {
                                    if (entry.modelIndex == state.modelIndex &&
                                        entry.sourceNode == channel.node &&
                                        entry.morphTargetCount > 0) {
                                        sampleChannel(channel, state.time,
                                                      morphWeightsFrame.data() +
                                                          entry.morphWeightOffset,
                                                      entry.morphTargetCount);
                                    }
                                }
                                break;
                            }
                        }
                    }
                    for (std::size_t i = 0; i < nodeCount; ++i) {
                        const math::Mat4 local = composeTrs(state.t[i], state.r[i], state.s[i]);
                        state.world[i] =
                            skeleton.nodes[i].parent >= 0
                                ? math::mul(state.world[static_cast<std::size_t>(
                                                skeleton.nodes[i].parent)],
                                            local)
                                : local;
                    }
                    for (std::size_t j = 0; j < skeleton.jointNodes.size(); ++j) {
                        jointMatricesCpu[state.jointBase + j] = math::mul(
                            state.modelMatrix,
                            math::mul(state.world[skeleton.jointNodes[j]],
                                      skeleton.inverseBind[j]));
                    }
                }
                const std::uint32_t animSlot = renderer->frameSlot();
                if (jointBuffer && !jointMatricesCpu.empty()) {
                    std::memcpy(static_cast<std::byte*>(jointBuffer->mapped()) +
                                    animSlot * jointMatricesCpu.size() * sizeof(math::Mat4),
                                jointMatricesCpu.data(),
                                jointMatricesCpu.size() * sizeof(math::Mat4));
                }
                if (morphWeightBuffer && !morphWeightsFrame.empty()) {
                    std::memcpy(static_cast<std::byte*>(morphWeightBuffer->mapped()) +
                                    animSlot * morphWeightsFrame.size() * sizeof(float),
                                morphWeightsFrame.data(),
                                morphWeightsFrame.size() * sizeof(float));
                }
            }

            REND_PROFILE_ZONE("CameraLightWrite");
            CameraData cameraData;
            cameraData.viewProj = camera.viewProj(viewWidth, viewHeight);
            {
                // Ray-generation basis for the traced primary pass: the
                // lookAt frame with right/up prescaled by the frustum.
                const math::Vec3 camFwd = camera.forward();
                const math::Vec3 camRight =
                    math::normalize(math::cross(camFwd, {0.0f, 1.0f, 0.0f}));
                const math::Vec3 camUp = math::cross(camRight, camFwd);
                const float tanHalf = std::tan(camera.fovDegrees * kPi / 360.0f);
                const float rayAspect =
                    viewHeight > 0 ? static_cast<float>(viewWidth) / viewHeight : 1.0f;
                // position.w = per-pixel ray-cone spread angle: how much a
                // pixel's footprint widens per unit distance (traced LOD).
                cameraData.position = {camera.position.x, camera.position.y, camera.position.z,
                                       viewHeight > 0 ? 2.0f * tanHalf / viewHeight : 0.001f};
                cameraData.rightAxis = {camRight.x * tanHalf * rayAspect,
                                        camRight.y * tanHalf * rayAspect,
                                        camRight.z * tanHalf * rayAspect, 0.0f};
                cameraData.upAxis = {camUp.x * tanHalf, camUp.y * tanHalf, camUp.z * tanHalf,
                                     0.0f};
                cameraData.forwardAxis = {camFwd.x, camFwd.y, camFwd.z, 0.0f};
                // LOD screen-size scale for the cull pass: pixels per
                // world unit at unit distance over the target pixel size.
                // Rides the push constants (baked into static recordings),
                // but its only inputs — viewport height and fov — change
                // exactly when the recordings rebuild anyway (resize).
                batch.lodFactor =
                    viewHeight > 0
                        ? static_cast<float>(viewHeight) / (2.0f * tanHalf * kLodTargetPixels)
                        : 0.0f;
            }
            std::memcpy(static_cast<std::byte*>(cameraBuffer->mapped()) +
                            renderer->frameSlot() * sizeof(CameraData),
                        &cameraData, sizeof(CameraData));
            LightData lightData;
            const math::Vec3 direction = sun.direction();
            const float aspect =
                viewHeight > 0 ? static_cast<float>(viewWidth) / viewHeight : 1.0f;
            if (sceneAabbDirty) {
                recomputeSceneAabb();
                sceneAabbDirty = false;
            }
            const CascadeFit fit =
                fitCascades(camera.position, camera.forward(), camera.fovDegrees, aspect,
                            direction, sceneMin, sceneMax, sun.shadowDistance);
            lightData.cascadeViewProj = fit.viewProj;
            lightData.splitDepths = fit.splits;
            lightData.direction = {direction.x, direction.y, direction.z};
            lightData.intensity = sun.intensity;
            lightData.color = sun.color;
            lightData.pcfRadius = sun.pcfRadius;
            lightData.biasBase = sun.biasBase;
            lightData.cascadeCount = batch.cascadeCount;
            lightData.debugTint = sun.debugTint ? 1u : 0u;
            lightData.rtShadows = (rtReady && sun.rtShadows) ? 1u : 0u;
            lightData.reflections = (rtReady && reflectionsTraced) ? kReflectionTraced
                                    : probeReady                  ? kReflectionProbe
                                                                  : kReflectionNone;
            if (scene && scene->fog.enabled) {
                const assetio::FogDesc& fog = scene->fog;
                lightData.fogBoxMin = {fog.position[0] - fog.size[0] * 0.5f,
                                       fog.position[1] - fog.size[1] * 0.5f,
                                       fog.position[2] - fog.size[2] * 0.5f, fog.density};
                lightData.fogBoxMax = {fog.position[0] + fog.size[0] * 0.5f,
                                       fog.position[1] + fog.size[1] * 0.5f,
                                       fog.position[2] + fog.size[2] * 0.5f, fog.anisotropy};
                lightData.fogColor = {fog.color[0], fog.color[1], fog.color[2], fog.steps};
            }
            // Sky/ambient/point lights: per-slot buffer data, so scripts
            // changing them per frame never touch the static recordings.
            // The global scale fades every authored intensity; the count
            // reflects post-scale intensities so a scale of 0 (daytime)
            // skips the shading loop entirely.
            lightData.pointLights = pointLights;
            std::uint32_t activeLights = 0;
            for (std::uint32_t i = 0; i < kMaxPointLights; ++i) {
                lightData.pointLights[i].colorIntensity[3] *= pointLightScale;
                if (lightData.pointLights[i].colorIntensity[3] > 0.0f) {
                    activeLights = i + 1;
                }
            }
            lightData.skyColor = {skyColor[0], skyColor[1], skyColor[2],
                                  static_cast<float>(activeLights)};
            lightData.ambientColor = {ambientColor[0], ambientColor[1], ambientColor[2],
                                      timeOfDay};
            std::memcpy(static_cast<std::byte*>(lightBuffer->mapped()) +
                            renderer->frameSlot() * sizeof(LightData),
                        &lightData, sizeof(LightData));
        }

        if (ui && viewWidth > 0 && viewHeight > 0) {
            REND_PROFILE_ZONE("BuildUi");
            statsRefreshTimer += deltaSeconds;
            if (statsRefreshTimer >= 0.25f) {
                statsRefreshTimer = 0.0f;
                std::memcpy(shownDrawCounts, lastDrawCounts, sizeof(shownDrawCounts));
            }
            const bool vsyncBefore = vsync;
            viewer::Ui::LoadingStatus loadingStatus{
                .hideScene = waitForTextures,
                .done = static_cast<std::uint32_t>(texStream.uploaded),
                .total = static_cast<std::uint32_t>(texStream.total),
            };
            const bool loadingVisible = texStream.active || waitForTextures;
            const viewer::Ui::CullStats cullStats{
                .drawsInView = lastDrawCounts[1] + lastDrawCounts[3],
                .drawsLive = lastDrawCounts[0],
                .triangles = (lastDrawCounts[4] + lastDrawCounts[5]) / 3,
                .occluded = lastDrawCounts[6],
            };
            // No culling section while traced primary owns the frame: the
            // cull dispatch is skipped there, so the counters would freeze
            // at their last raster values.
            ui->buildFrame(viewWidth, viewHeight, deltaSeconds, &vsync,
                           loadingVisible ? &loadingStatus : nullptr,
                           batch.cullPipeline && !batch.rtPrimary ? &cullStats : nullptr);
            if (vsync != vsyncBefore) {
                // The preference lands on the next swapchain build; forcing
                // a same-size resize triggers that recreate (and the static
                // command-buffer rebuild that comes with it).
                swapchain->setVsync(vsync);
                renderer->resize(viewWidth, viewHeight);
                log::info("VSync {}", vsync ? "on" : "off");
            }
            if (drawScene && !waitForTextures) {
                // Sun & shadow tuning; changes land in the light buffer on
                // the next frame's write.
                // Below the debug panel (FPS + graph + VSync) in the corner.
                // Below the debug panel, which grew GPU-memory and culling
                // sections (screenshots 055/039 caught earlier overlaps of
                // exactly this kind — keep this below the panel's bottom).
                ImGui::SetNextWindowPos(ImVec2(8.0f, 420.0f), ImGuiCond_FirstUseEver);
                ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
                // One shadow choice, built from the device's offer list.
                // Ray traced additionally needs the BVH the viewer built.
                const gpu::ShadowTechnique active = (rtReady && sun.rtShadows)
                                                        ? gpu::ShadowTechnique::RayTraced
                                                        : gpu::ShadowTechnique::CascadedShadowMaps;
                if (batch.rtPrimaryPipeline) {
                    // Recording differs between the two modes, so flipping
                    // drops any static command buffers (lazy rebuild).
                    const char* primaryNames[] = {"Raster", "Ray traced"};
                    int primary = batch.rtPrimary ? 1 : 0;
                    if (ImGui::Combo("Primary rays", &primary, primaryNames, 2)) {
                        batch.rtPrimary = primary == 1;
                        renderer->invalidateStaticRecordings();
                    }
                }
                if (!batch.rtPrimary &&
                    ImGui::BeginCombo("Shadows", gpu::shadowTechniqueName(active))) {
                    // Hidden while primary rays are traced: that path fires
                    // its shadow rays from the hit points regardless.
                    for (gpu::ShadowTechnique offer : shadowOffers) {
                        if (offer == gpu::ShadowTechnique::RayTraced && !rtReady) {
                            continue;
                        }
                        if (ImGui::Selectable(gpu::shadowTechniqueName(offer), offer == active)) {
                            sun.rtShadows = offer == gpu::ShadowTechnique::RayTraced;
                        }
                    }
                    ImGui::EndCombo();
                }
                // Reflection technique for reflective-tagged objects, from
                // the device's offer list. Hidden while primary rays are
                // traced — that path reflects everything for real, the
                // per-object choice is superseded (same as Shadows above).
                const gpu::ReflectionTechnique activeReflection =
                    (rtReady && reflectionsTraced) ? gpu::ReflectionTechnique::RayTraced
                                                   : gpu::ReflectionTechnique::ReflectionProbe;
                if (!batch.rtPrimary && (probeReady || rtReady) &&
                    ImGui::BeginCombo("Reflections",
                                      gpu::reflectionTechniqueName(activeReflection))) {
                    for (gpu::ReflectionTechnique offer : reflectionOffers) {
                        if (offer == gpu::ReflectionTechnique::ReflectionProbe && !probeReady) {
                            continue;
                        }
                        if (offer == gpu::ReflectionTechnique::RayTraced && !rtReady) {
                            continue;
                        }
                        if (ImGui::Selectable(gpu::reflectionTechniqueName(offer),
                                              offer == activeReflection)) {
                            reflectionsTraced = offer == gpu::ReflectionTechnique::RayTraced;
                        }
                    }
                    ImGui::EndCombo();
                }
                if (!batch.rtPrimary && batch.cullPipeline) {
                    if (ImGui::Checkbox("Frustum culling", &frustumCull)) {
                        // The flag rides the cull dispatch's push
                        // constants, which static recordings bake.
                        batch.cullFlags = (frustumCull ? 1u : 0u) | (lodSelect ? 2u : 0u) |
                                          (occlusionCull && occlusionPipeline ? 4u : 0u);
                        renderer->invalidateStaticRecordings();
                        log::info("Frustum culling {}", frustumCull ? "on" : "off");
                    }
                    if (ImGui::Checkbox("LOD selection", &lodSelect)) {
                        batch.cullFlags = (frustumCull ? 1u : 0u) | (lodSelect ? 2u : 0u) |
                                          (occlusionCull && occlusionPipeline ? 4u : 0u);
                        renderer->invalidateStaticRecordings();
                        log::info("LOD selection {}", lodSelect ? "on" : "off");
                    }
                    if (occlusionPipeline &&
                        ImGui::Checkbox("Occlusion culling", &occlusionCull)) {
                        batch.cullFlags = (frustumCull ? 1u : 0u) | (lodSelect ? 2u : 0u) |
                                          (occlusionCull ? 4u : 0u);
                        renderer->invalidateStaticRecordings();
                        log::info("Occlusion culling {}", occlusionCull ? "on" : "off");
                    }
                    if (occlusionDebugPipeline &&
                        ImGui::Checkbox("Show occlusion boxes", &showOcclusionBoxes)) {
                        // Pipeline presence is baked into static
                        // recordings: one rebuild per flip, zero
                        // per-frame CPU cost afterwards.
                        batch.occlusionDebugPipeline =
                            showOcclusionBoxes ? occlusionDebugPipeline.get() : nullptr;
                        batch.occlusionInstanceDebugPipeline =
                            showOcclusionBoxes ? occlusionInstanceDebugPipeline.get() : nullptr;
                        renderer->invalidateStaticRecordings();
                        log::info("Occlusion box overlay {}", showOcclusionBoxes ? "on" : "off");
                    }
                    ImGui::Text("Draws: %u in view / %u live / %u table", shownDrawCounts[1],
                                shownDrawCounts[0], batch.drawCount);
                    ImGui::Text("Transparent draws: %u", shownDrawCounts[3]);
                    ImGui::Text("Partial instance rows: %u", shownDrawCounts[2]);
                    // Free stat: the cull pass already counts emitted
                    // indices; this readback rides the existing fenced
                    // count-buffer memcpy, so it costs no GPU time at all.
                    ImGui::Text("Triangles: %u (%u transparent)",
                                (shownDrawCounts[4] + shownDrawCounts[5]) / 3,
                                shownDrawCounts[5] / 3);
                    ImGui::Text("Occluded: %u", shownDrawCounts[6]);
                }
                if (ImGui::TreeNode("Advanced")) {
                    ImGui::SliderFloat("Azimuth", &sun.azimuthDeg, -180.0f, 180.0f, "%.0f deg");
                    ImGui::SliderFloat("Elevation", &sun.elevationDeg, 10.0f, 90.0f, "%.0f deg");
                    ImGui::SliderFloat("Intensity", &sun.intensity, 0.0f, 4.0f, "%.2f");
                    ImGui::ColorEdit3("Color", sun.color.data());
                    ImGui::SliderFloat("PCF radius", &sun.pcfRadius, 0.0f, 4.0f, "%.1f texels");
                    ImGui::SliderFloat("Bias", &sun.biasBase, 0.0002f, 0.0060f, "%.4f");
                    ImGui::SliderFloat("Shadow dist", &sun.shadowDistance, 5.0f, 150.0f, "%.0f m");
                    ImGui::Checkbox("Show cascades", &sun.debugTint);
                    ImGui::TreePop();
                }
                ImGui::End();
            }
        }

        // Wait-mode loading holds the scene entirely off the frame (the
        // batch is withheld, so only the clear color and the fullscreen
        // ImGui cover render); streaming mode draws it from the first
        // frame, white textures popping to real ones as they land.
        const bool sceneVisible = drawScene && !waitForTextures;
        if (auto r = renderer->drawFrame(sceneVisible ? &batch : nullptr);
            !r) {
            log::error("Frame failed: {}", r.error().message);
            running = false;
        }
        REND_PROFILE_FRAME();

        ++frame;
        if (frame % kReportInterval == 0) {
            report(kReportInterval);
        }
        if (benchFrames != 0 && frame >= benchFrames) {
            report(frame % kReportInterval);
            log::info("Benchmark complete after {} frames", frame);
            running = false;
        }
    }

    log::info("Shutting down");
    // Python first: scripts may still be pushing commands; after Stop the
    // queue is quiet. (The DLL stays loaded — CPython dislikes unloading.)
    if (pyhostStop) {
        pyhostStop();
    }
    // The loader references importers + the claim set; stop it before any
    // teardown. Pending requests and undelivered prepared loads just drop.
    {
        std::lock_guard lock(loaderMutex);
        loaderQuit = true;
    }
    loaderWake.notify_one();
    loaderThread.join();
    // Streaming decode workers (quit mid-load): each finishes its current
    // file, sees the flag, and exits; undelivered decodes just drop.
    texStream.quit = true;
    for (std::thread& worker : texStream.workers) {
        worker.join();
    }
    texStream.workers.clear();
    renderer->waitIdle();
    ui.reset(); // ImGui's Vulkan objects go while the device is idle and alive
    // The swapchain goes first: destroying it retires presents that are still
    // waiting on the frame renderer's per-image semaphores, which the renderer
    // then destroys. It also owns the surface, so it must precede the instance.
    swapchain.reset();
    renderer.reset();
    target.reset();
    backend->shutdown();
    return 0;
}

#include "rend/assetio/scene_loader.h"
#include "rend/assetio/texture_loader.h"
#include "rend/core/log.h"
#include "rend/core/math.h"
#include "rend/core/paths.h"
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

#include "ui.h"

#include <imgui.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <format>
#include <cstddef>
#include <cstring>
#include <optional>
#include <unordered_map>

using namespace rend;

namespace {

// Where a mesh landed in the geometry pool; becomes one indirect entry.
struct GeometryLocation {
    gpu::BufferSlice vertices;
    gpu::BufferSlice indices;
    std::uint32_t indexCount = 0;
    std::uint32_t materialIndex = 0;
};

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

constexpr std::uint32_t kShadowMapSize = 2048;
constexpr std::uint32_t kShadowCascades = 4;
constexpr float kPi = 3.14159265358979323846f;

// Reflection probe capture: face resolution and format of the load-time
// cubemap, plus the six extra camera/light buffer regions the capture pass
// indexes with its push-constant slot (appended after the per-frame ones,
// so scene.hlsl needs no changes to render probe faces).
constexpr std::uint32_t kProbeFaceSize = 256;
constexpr std::uint32_t kProbeFormat = 43; // VK_FORMAT_R8G8B8A8_SRGB
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
};
static_assert(sizeof(LightData) == 336);

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
    std::uint64_t benchFrames = 0; // non-zero: exit after N frames with a report
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
        auto sceneResult = assetio::loadScene(scenePath, assetio::ImporterRegistry::withBuiltins());
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
                for (std::size_t v = 0; v + 2 < mesh.positions.size(); v += 3) {
                    const float x = mesh.positions[v], y = mesh.positions[v + 1],
                                z = mesh.positions[v + 2];
                    mesh.positions[v] = m[0] * x + m[4] * y + m[8] * z + m[12];
                    mesh.positions[v + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
                    mesh.positions[v + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
                }
                for (std::size_t v = 0; v + 2 < mesh.normals.size(); v += 3) {
                    const float x = mesh.normals[v], y = mesh.normals[v + 1],
                                z = mesh.normals[v + 2];
                    const math::Vec3 n = math::normalize({m[0] * x + m[4] * y + m[8] * z,
                                                          m[1] * x + m[5] * y + m[9] * z,
                                                          m[2] * x + m[6] * y + m[10] * z});
                    mesh.normals[v] = n.x;
                    mesh.normals[v + 1] = n.y;
                    mesh.normals[v + 2] = n.z;
                }
                // Morph deltas are direction-like: rotate/scale, no
                // translation, and position deltas keep their magnitudes.
                for (auto& target : mesh.morphTargets) {
                    for (auto* deltas : {&target.positionDeltas, &target.normalDeltas}) {
                        for (std::size_t v = 0; v + 2 < deltas->size(); v += 3) {
                            const float x = (*deltas)[v], y = (*deltas)[v + 1],
                                        z = (*deltas)[v + 2];
                            (*deltas)[v] = m[0] * x + m[4] * y + m[8] * z;
                            (*deltas)[v + 1] = m[1] * x + m[5] * y + m[9] * z;
                            (*deltas)[v + 2] = m[2] * x + m[6] * y + m[10] * z;
                        }
                    }
                }
            }
        }
    } else {
        log::info("No scene file given "
                  "(usage: viewer [--debug] [--novsync] [--static] [--bench N] "
                  "[--draw-mode count|indirect|direct] <scene.xml>)");
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
    std::vector<gpu::DrawIndexedIndirect> draws; // outlives the loop: Direct mode records from it
    std::unique_ptr<gpu::Buffer> objectBuffer;
    std::unique_ptr<gpu::DescriptorTable> descriptorTable;
    std::vector<std::unique_ptr<gpu::Image>> textures;
    std::vector<GeometryLocation> geometry;
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
    if (scene) {
        const auto start = std::chrono::steady_clock::now();
        // With ray tracing, the geometry pool doubles as the BLAS build
        // input, which needs device-address usage.
        // Storage usage always: the skinning pass poses vertices in place;
        // on RT devices the traced primary pass also reads hit triangles
        // straight out of the pool (descriptor binding 11).
        auto poolResult = gpu::MemoryPool::create(
            *device, 128ull * 1024 * 1024,
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
        auto transfer = std::move(transferResult).value();

        // Slot 0 of the bindless texture array is a 1x1 white fallback so
        // untextured materials sample neutrally; index 0 doubles as "no
        // normal / no metallic-roughness map".
        struct TextureRequest {
            std::filesystem::path path;
            bool srgb = true; // normal/MR maps decode linear (UNORM)
        };
        std::vector<TextureRequest> texturePaths{{}};
        std::unordered_map<std::string, std::uint32_t> textureSlotByPath;
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
        std::vector<ObjectData> objectData;

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

                for (std::size_t v = 0; v + 2 < mesh.positions.size(); v += 3) {
                    sceneMin.x = std::min(sceneMin.x, mesh.positions[v]);
                    sceneMin.y = std::min(sceneMin.y, mesh.positions[v + 1]);
                    sceneMin.z = std::min(sceneMin.z, mesh.positions[v + 2]);
                    sceneMax.x = std::max(sceneMax.x, mesh.positions[v]);
                    sceneMax.y = std::max(sceneMax.y, mesh.positions[v + 1]);
                    sceneMax.z = std::max(sceneMax.z, mesh.positions[v + 2]);
                }

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
        draws.reserve(geometry.size());
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
        // the current slot's instanceCounts every frame (the milestone-7
        // load/unload toggle) while the other slot's region is in flight.
        const std::uint64_t indirectRegion = draws.size() * sizeof(gpu::DrawIndexedIndirect);
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
                        slotDraws.data(), indirectRegion);
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

        // Draw-count buffer for IndirectCount mode: one uint32 per frame
        // slot, written by the cull pass (zeroed via fill, incremented by
        // the shader) and read by vkCmdDrawIndexedIndirectCount.
        auto countResult = gpu::Buffer::create(
            *device, {
                         .size = sizeof(std::uint32_t) * gpu::FrameRenderer::kFramesInFlight,
                         .usage = gpu::kUsageIndirect | gpu::kUsageStorage |
                                  gpu::kUsageTransferDst,
                         .location = gpu::MemoryLocation::DeviceLocal,
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
                         .size = objectData.size() * sizeof(ObjectData),
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
                             .size = geometryInfo.size() * sizeof(GeometryInfo),
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

        if (auto flushed = transfer->flush(); !flushed) {
            log::error("Geometry upload failed: {}", flushed.error().message);
            return 1;
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
                const gpu::AccelerationStructure::Instance blasInstance{.blas = blas.get()};
                auto tlasResult = gpu::AccelerationStructure::buildTopLevel(
                    *device, {&blasInstance, 1}, animatedBvh);
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
        descriptorTable->writeObjectBuffer(objectBuffer->handle(),
                                           objectData.size() * sizeof(ObjectData));
        descriptorTable->writeStorageBuffer(3, indirectBuffer->handle(), indirectBuffer->size());
        descriptorTable->writeStorageBuffer(4, compactedBuffer->handle(),
                                            compactedBuffer->size());
        descriptorTable->writeStorageBuffer(5, countBuffer->handle(), countBuffer->size());
        descriptorTable->writeStorageBuffer(6, cameraBuffer->handle(), cameraBuffer->size());
        descriptorTable->writeStorageBuffer(7, lightBuffer->handle(), lightBuffer->size());
        if (!animatedMeshes.empty()) {
            descriptorTable->writeStorageBuffer(13, geometryPool->buffer().handle(),
                                                geometryPool->buffer().size());
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
        }

        auto uploaderResult = gpu::TextureUploader::create(*device);
        if (!uploaderResult) {
            log::error("Texture uploader creation failed: {}", uploaderResult.error().message);
            return 1;
        }
        auto uploader = std::move(uploaderResult).value();

        std::uint64_t texelBytes = 0;
        for (std::size_t i = 0; i < texturePaths.size(); ++i) {
            Result<std::unique_ptr<gpu::Image>> uploaded = [&]() {
                if (i == 0) {
                    const std::uint8_t white[4] = {255, 255, 255, 255};
                    return uploader->upload(1, 1, white);
                }
                auto decoded = assetio::loadTexture(texturePaths[i].path);
                if (!decoded) {
                    return Result<std::unique_ptr<gpu::Image>>{decoded.error()};
                }
                const auto& t = decoded.value();
                texelBytes += t.rgba.size();
                return uploader->upload(t.width, t.height, t.rgba.data(),
                                        texturePaths[i].srgb);
            }();
            if (!uploaded) {
                log::error("Texture {} failed: {}", texturePaths[i].path.filename().string(),
                           uploaded.error().message);
                return 1;
            }
            descriptorTable->writeTexture(static_cast<std::uint32_t>(i),
                                          uploaded.value()->view());
            textures.push_back(std::move(uploaded).value());
        }
        const auto texMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - texStart)
                               .count();
        log::info("Textures ready in {} ms: {} images ({:.1f} MiB decoded), mipmapped, bindless",
                  texMs, textures.size(), static_cast<double>(texelBytes) / (1024.0 * 1024.0));

        // Reflection probe capture: render the scene's draw stream into a
        // small cubemap once at load — the raster tier reflective objects
        // sample. Static by nature: animated meshes bake at the bind pose,
        // lighting never re-captures, no parallax correction (v1 gaps by
        // design; ray traced reflections are the exact tier above).
        {
            const auto probeStart = std::chrono::steady_clock::now();
            // Probe position: first <ReflectionProbe> in the scene XML,
            // else the scene AABB's center.
            math::Vec3 probePos = vmul(vadd(sceneMin, sceneMax), 0.5f);
            if (!scene->reflectionProbes.empty()) {
                const auto& p = scene->reflectionProbes.front().position;
                probePos = {p[0], p[1], p[2]};
            }

            // Six capture camera regions after the per-frame ones: 90-degree
            // faces in the spec's cube texel basis (see kProbeFaceBases).
            const math::Mat4 proj = math::perspective(kPi * 0.5f, 1.0f, 0.05f, 300.0f);
            for (std::uint32_t face = 0; face < kProbeFaces; ++face) {
                const ProbeFaceBasis& basis = kProbeFaceBases[face];
                CameraData faceCamera;
                faceCamera.viewProj =
                    math::mul(proj, faceView(probePos, basis.right, basis.up, basis.forward));
                faceCamera.position = {probePos.x, probePos.y, probePos.z, 0.001f};
                faceCamera.rightAxis = {basis.right.x, basis.right.y, basis.right.z, 0.0f};
                faceCamera.upAxis = {basis.up.x, basis.up.y, basis.up.z, 0.0f};
                faceCamera.forwardAxis = {basis.forward.x, basis.forward.y, basis.forward.z,
                                          0.0f};
                std::memcpy(static_cast<std::byte*>(cameraBuffer->mapped()) +
                                (gpu::FrameRenderer::kFramesInFlight + face) *
                                    sizeof(CameraData),
                            &faceCamera, sizeof(CameraData));
                // Matching light regions: sun color/direction but no shadow
                // sampling (cascadeCount 0, no maps exist yet) and no
                // reflections — the capture sees reflective objects as
                // plain surfaces instead of sampling the probe being made.
                LightData faceLight;
                const SunControls captureSun = SunControls::fromLight(
                    scene->lights.empty() ? assetio::LightDesc{} : scene->lights.front());
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
                                     .drawCount = static_cast<std::uint32_t>(draws.size()),
                                     .descriptors = descriptorTable->set(),
                                     .pipeline = probePipeline.get(),
                                     .format = kProbeFormat,
                                     .faceSize = kProbeFaceSize,
                                     .cameraSlotBase = gpu::FrameRenderer::kFramesInFlight,
                                 });
                    if (probeResult) {
                        probeImage = std::move(probeResult).value();
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
        }
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
    auto vertexResult = gpu::Shader::createFromFile(*device, shaderDir / "triangle.vert.spv");
    if (!vertexResult) {
        log::error("{}", vertexResult.error().message);
        return 1;
    }
    auto fragmentResult = gpu::Shader::createFromFile(*device, shaderDir / "triangle.frag.spv");
    if (!fragmentResult) {
        log::error("{}", fragmentResult.error().message);
        return 1;
    }
    auto vertexShader = std::move(vertexResult).value();
    auto fragmentShader = std::move(fragmentResult).value();

    auto pipelineResult = gpu::Pipeline::createGraphics(*device,
                                                        {
                                                            .vertexShader = vertexShader.get(),
                                                            .fragmentShader = fragmentShader.get(),
                                                            .colorFormat = swapchain->imageFormat(),
                                                        });
    if (!pipelineResult) {
        log::error("Pipeline creation failed: {}", pipelineResult.error().message);
        return 1;
    }
    auto pipeline = std::move(pipelineResult).value();

    // Scene pass pipeline: interleaved vertex input from the geometry pool,
    // depth-tested, camera via push constant.
    std::unique_ptr<gpu::Pipeline> scenePipeline;
    std::unique_ptr<gpu::Pipeline> cullPipeline;
    std::unique_ptr<gpu::Pipeline> shadowPipeline;
    std::unique_ptr<gpu::Pipeline> skinPipeline;
    std::unique_ptr<gpu::Shader> sceneVert, sceneFrag, cullShader, shadowVert, shadowFrag,
        skinShader;
    if (scene) {
        auto vertResult = gpu::Shader::createFromFile(*device, shaderDir / "scene.vert.spv");
        // Scene-shipped fragment override (scene-local looks like toon)
        // beats the built-ins; else the RT variant traces shadow rays
        // inline — only loadable where the device enabled RayQuery (the
        // SPIR-V declares the capability). Override applies scene-wide for
        // now (single pipeline); per-model pipelines arrive with the
        // renderer layer.
        std::filesystem::path fragmentPath =
            shaderDir / (rtReady ? "scene_rt.frag.spv" : "scene.frag.spv");
        for (const auto& model : scene->models) {
            if (!model.desc.fragmentShaderPath.empty()) {
                if (std::filesystem::exists(model.desc.fragmentShaderPath)) {
                    fragmentPath = model.desc.fragmentShaderPath;
                    log::info("Scene fragment override: {}", fragmentPath.string());
                } else {
                    log::warn("Scene fragment override missing, using built-in: {}",
                              model.desc.fragmentShaderPath.string());
                }
                break;
            }
        }
        auto fragResult = gpu::Shader::createFromFile(*device, fragmentPath);
        if (!vertResult || !fragResult) {
            log::error("{}", (!vertResult ? vertResult : fragResult).error().message);
            return 1;
        }
        sceneVert = std::move(vertResult).value();
        sceneFrag = std::move(fragResult).value();
        auto sceneResult = gpu::Pipeline::createGraphics(
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
                     });
        if (!sceneResult) {
            log::error("Scene pipeline creation failed: {}", sceneResult.error().message);
            return 1;
        }
        scenePipeline = std::move(sceneResult).value();

        // The compaction pass (IndirectCount mode only). A failure here is
        // not fatal: the draw-mode ladder just skips to Indirect.
        auto cullShaderResult = gpu::Shader::createFromFile(*device, shaderDir / "cull.comp.spv");
        if (cullShaderResult) {
            cullShader = std::move(cullShaderResult).value();
            auto cullResult = gpu::Pipeline::createCompute(
                *device, {
                             .shader = cullShader.get(),
                             .descriptorLayout = descriptorTable->layout(),
                             .pushConstantBytes = 2 * sizeof(std::uint32_t),
                         });
            if (cullResult) {
                cullPipeline = std::move(cullResult).value();
            } else {
                log::warn("Cull pipeline unavailable: {}", cullResult.error().message);
            }
        } else {
            log::warn("Cull shader unavailable: {}", cullShaderResult.error().message);
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

    // With a scene, every frame is the indirect batch over the geometry
    // pool; without one, the milestone-6 triangle stays as the fallback.
    const bool drawScene = scenePipeline && indirectBuffer && !geometry.empty();
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
        batch.indirectRegionStride = geometry.size() * sizeof(gpu::DrawIndexedIndirect);
        if (batch.mode == gpu::DrawSubmitMode::IndirectCount) {
            // The GPU draws what the cull pass compacted, not the templates.
            batch.indirect = compactedBuffer->handle();
            batch.cullPipeline = cullPipeline.get();
        } else {
            batch.indirect = indirectBuffer->handle();
        }
        batch.count = countBuffer->handle();
        batch.countRegionStride = sizeof(std::uint32_t);
        batch.cpuDraws = draws.data();
        batch.descriptors = descriptorTable->set();
        if (shadowPipeline && !shadowMaps.empty() &&
            (scene->lights.empty() || scene->lights.front().castsShadows)) {
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
                    batch.refitTlas = tlas.get();
                    batch.refitGeometries = refitGeometries;
                }
            }
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
        log::info("[{}] {} frames in {:.2f} s = {:.0f} fps | record {:.1f} us/frame{}",
                  renderer->staticRecording() ? "static" : "rerecord", windowFrames, seconds,
                  windowFrames / seconds,
                  stats.frames > 0 ? static_cast<double>(stats.recordMicros) / stats.frames : 0.0,
                  stats.prerecords > 0 ? std::format(" | {} prerecords", stats.prerecords) : "");
    };

    FlyCamera camera;
    SunControls sun;
    // Reflection technique for reflective-tagged objects in the raster
    // path; defaults to the best offer (traced where available, so nothing
    // visually regresses vs. the per-object RT milestone).
    bool reflectionsTraced = false;
    if (scene) {
        camera = FlyCamera::fromScene(scene->camera);
        sun = SunControls::fromLight(scene->lights.empty() ? assetio::LightDesc{}
                                                           : scene->lights.front());
        sun.rtShadows = rtFromStart && rtReady;
        reflectionsTraced = rtReady && !forceProbeReflections;
        log::info("Sun: azimuth {:.0f}, elevation {:.0f}, intensity {:.2f}, shadows {}",
                  sun.azimuthDeg, sun.elevationDeg, sun.intensity,
                  batch.shadowPipeline ? "on" : "off");
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
        for (const auto& event : backend->pumpEvents()) {
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

            // Publish this frame's camera into the slot's region: safe to
            // write once the slot's previous submission retired.
            if (auto r = renderer->waitFrameSlot(); !r) {
                log::error("Frame failed: {}", r.error().message);
                break;
            }
            // Animation playback: sample every channel, rebuild node worlds
            // and joint matrices, and write this slot's regions (safe after
            // waitFrameSlot). The skinning pass consumes them this frame.
            if (batch.skinPipeline && !animatedStates.empty()) {
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
            }
            std::memcpy(static_cast<std::byte*>(cameraBuffer->mapped()) +
                            renderer->frameSlot() * sizeof(CameraData),
                        &cameraData, sizeof(CameraData));
            LightData lightData;
            const math::Vec3 direction = sun.direction();
            const float aspect =
                viewHeight > 0 ? static_cast<float>(viewWidth) / viewHeight : 1.0f;
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
            std::memcpy(static_cast<std::byte*>(lightBuffer->mapped()) +
                            renderer->frameSlot() * sizeof(LightData),
                        &lightData, sizeof(LightData));
        }

        if (ui && viewWidth > 0 && viewHeight > 0) {
            ui->buildFrame(viewWidth, viewHeight, deltaSeconds);
            if (drawScene) {
                // Sun & shadow tuning; changes land in the light buffer on
                // the next frame's write.
                ImGui::SetNextWindowPos(ImVec2(8.0f, 40.0f), ImGuiCond_FirstUseEver);
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

        if (auto r = renderer->drawFrame(drawScene ? *scenePipeline : *pipeline,
                                         drawScene ? &batch : nullptr);
            !r) {
            log::error("Frame failed: {}", r.error().message);
            running = false;
        }

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
    renderer->waitIdle();
    ui.reset(); // ImGui's Vulkan objects go while the device is idle and alive
    // The swapchain goes first: destroying it retires presents that are still
    // waiting on the frame renderer's per-image semaphores, which the renderer
    // then destroys. It also owns the surface, so it must precede the instance.
    swapchain.reset();
    renderer.reset();
    pipeline.reset();
    fragmentShader.reset();
    vertexShader.reset();
    target.reset();
    backend->shutdown();
    return 0;
}

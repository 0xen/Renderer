#pragma once

#include "rend/core/result.h"
#include "rend/gpu/acceleration_structure.h"
#include "rend/gpu/pipeline.h" // kFormat* constants for kGBufferFormats

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

typedef struct VkCommandPool_T* VkCommandPool;
typedef struct VkCommandBuffer_T* VkCommandBuffer;
typedef struct VkSemaphore_T* VkSemaphore;
typedef struct VkFence_T* VkFence;
typedef struct VkBuffer_T* VkBuffer;
typedef struct VkDescriptorSet_T* VkDescriptorSet;

namespace rend::gpu {

class DescriptorTable;
class Device;
class Image;
class Pipeline;
class Swapchain;

// Mirrors VkDrawIndexedIndirectCommand so callers can fill indirect
// buffers without Vulkan headers.
struct DrawIndexedIndirect {
    std::uint32_t indexCount = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t firstIndex = 0;
    std::int32_t vertexOffset = 0;
    std::uint32_t firstInstance = 0;
};
static_assert(sizeof(DrawIndexedIndirect) == 20);

// How a DrawBatch reaches the GPU, best first. The caller picks the best
// mode the device's enabled features allow (Device::isEnabled) — the gpu
// layer executes what it is told and never chooses policy.
enum class DrawSubmitMode {
    // vkCmdDrawIndexedIndirectCount: the GPU also reads the draw count, so
    // a compacted entry list needs no CPU round trip. Needs
    // Feature::DrawIndirectCount (+ the Indirect requirements).
    IndirectCount,
    // vkCmdDrawIndexedIndirect over all drawCount entries; hidden objects
    // are instanceCount-0 entries. Needs Feature::MultiDrawIndirect and
    // Feature::DrawIndirectFirstInstance.
    Indirect,
    // One vkCmdDrawIndexed per entry, read from cpuDraws on the CPU at
    // record time. Works on any Vulkan device; visibility changes need a
    // re-record (static recordings must be invalidated), so this is the
    // fallback of last resort.
    Direct,
};

// One indirect-draw submission over the geometry pool: the pool's buffer
// is bound once as vertex + index source, then every entry in the indirect
// buffer draws by offset. Adding/removing objects only rewrites entries.
struct DrawBatch {
    VkBuffer geometry = nullptr; // bound at offset 0 as VB and IB (uint32 indices)
    VkBuffer indirect = nullptr; // DrawIndexedIndirect[drawCount]
    std::uint32_t drawCount = 0;
    DrawSubmitMode mode = DrawSubmitMode::Indirect;
    // IndirectCount mode: buffer holding the uint32 draw count, one region
    // per frame slot (countRegionStride apart); drawCount caps it.
    VkBuffer count = nullptr;
    std::uint64_t countRegionStride = 0;
    // Direct mode: CPU-side copy of the drawCount entries.
    const DrawIndexedIndirect* cpuDraws = nullptr;
    // Shadow passes (optional): the same draw stream rendered depth-only
    // from the light's point of view into each cascade map before the main
    // pass, which then samples them (descriptor bindings 7-9). The light
    // matrices come from the per-slot light buffer, so static recordings
    // survive a moving sun; the cascade index rides the push constants.
    static constexpr std::uint32_t kMaxShadowCascades = 4;
    const Pipeline* shadowPipeline = nullptr;
    std::array<const Image*, kMaxShadowCascades> shadowCascades{}; // caller-owned
    std::uint32_t cascadeCount = 0;
    // GPU compaction (IndirectCount mode only): a compute pipeline whose
    // shader reads the draw templates (descriptor binding 3), appends
    // visible entries to `indirect` (binding 4) and counts them into
    // `count` (binding 5, eight uint32 per slot: [0] visibility-only for
    // the shadow passes, [1] frustum-culled opaques for the scene pass,
    // [2] the scratch-row allocator for per-instance culling, [3]
    // frustum-culled transparents for the blend pass, [4]/[5] indices
    // emitted to the opaque/transparent streams — stats). Recorded before
    // the render pass: zero the slot's counters, dispatch one thread per
    // template, barrier to the indirect + vertex reads. Null = none.
    const Pipeline* cullPipeline = nullptr;
    // Frustum-culled draw stream (binding 21) the MAIN pass draws in
    // IndirectCount mode; `indirect` keeps the visibility-only stream the
    // shadow passes draw (casters outside the camera frustum still cast).
    // Null = the main pass draws `indirect` too.
    VkBuffer sceneIndirect = nullptr;
    // Sky pass (optional): after the opaque draws, one fullscreen triangle
    // at the far plane paints the light buffer's per-slot skyColor over
    // the pixels no geometry covered (depth test LESS_OR_EQUAL, write
    // off), making the background per-frame dynamic even under static
    // recordings — the real clear value is baked at prerecord time.
    // Drawn before the transparency pass so glass blends over it. Null =
    // the baked clear color stays the background.
    const Pipeline* skyPipeline = nullptr;
    // Transparency pass (IndirectCount mode only): the cull shader routes
    // transparent-flagged entries into this stream (binding 24) instead of
    // sceneIndirect; after the opaque scene draw, the blend pipeline draws
    // it inside the same rendering pass (depth test on, depth write off).
    // Either null = no transparency pass (transparents then simply never
    // reach the transparent stream, or draw opaquely in lower tiers).
    VkBuffer transparentIndirect = nullptr;
    const Pipeline* transparentPipeline = nullptr;
    // Cull-shader flags pushed with the dispatch (bit 0 = frustum culling
    // against binding 22's per-object AABBs, bit 1 = LOD selection from
    // binding 25's per-entry tables, bit 2 = occlusion culling against
    // binding 26's proxy-pass visibility). Baked into static recordings —
    // invalidate them after flipping.
    std::uint32_t cullFlags = 0;
    // Occlusion proxy pass (IndirectCount mode, cullFlags bit 2): after
    // the transparency draw, every template's world AABB is rasterized as
    // an instanced unindexed cube (36 verts x drawCount) against the
    // frame's finished depth — test only, color masked — and surviving
    // fragments mark occlusionVisibility's current-slot region (binding
    // 26), which the NEXT frame's cull dispatch reads. The region is
    // zeroed alongside the count fill. Null pipeline = no proxy pass.
    const Pipeline* occlusionPipeline = nullptr;
    VkBuffer occlusionVisibility = nullptr;
    std::uint64_t occlusionRegionStride = 0; // bytes per frame-slot region
    // Per-INSTANCE proxy pass (runtime instanced models): drawn right
    // after the per-entry pass with the same push constants —
    // 36 verts x occlusionInstanceRows canonical rows; live local-mode
    // rows rasterize their transformed local box and mark the visibility
    // region's instance slots (entry capacity + row), which the next
    // frame's cull reads in its per-instance test. The visibility region
    // stride must cover entry capacity + occlusionInstanceRows slots.
    // Null pipeline or 0 rows = instanced entries are never occlusion-
    // culled (pre-existing behavior).
    const Pipeline* occlusionInstancePipeline = nullptr;
    std::uint32_t occlusionInstanceRows = 0;
    // Incremental GPU OBB refinement (IndirectCount mode): a small
    // dispatch baked just before the cull dispatch that claims the next
    // obbRefineGroups entries from a GPU-side counter (binding 33's
    // header) and fits each a PCA-oriented bounding box from its pooled
    // vertices, flipping the row's ready flag only when it beats the
    // AABB's volume. Self-terminating: once the counter passes drawCount
    // every dispatch exits immediately — the scene renders with AABBs
    // from frame 0 and the boxes tighten over the first seconds with no
    // recording or CPU involvement. Null = no refinement (binding 33
    // rows stay unready; consumers keep the AABBs forever).
    const Pipeline* obbRefinePipeline = nullptr;
    std::uint32_t obbRefineGroups = 8; // entries refined per frame
    // Debug overlay drawing the SAME proxy boxes as translucent color
    // (occlusionDebug pipeline state, proxy VS + debug PS). Drawn after
    // the proxy pass regardless of cullFlags bit 2 so the boxes can be
    // inspected with occlusion culling itself toggled off. Baked into
    // static recordings — invalidate after changing. Null = no overlay.
    // The instance variant draws the per-instance boxes the same way.
    const Pipeline* occlusionDebugPipeline = nullptr;
    const Pipeline* occlusionInstanceDebugPipeline = nullptr;
    // LOD screen-size scale pushed with the dispatch: pixels per world
    // unit at unit distance over the target error in pixels. 0 keeps
    // every entry at full detail. Depends only on the viewport height and
    // vertical fov, so the resize path's recording rebuild refreshes it.
    float lodFactor = 0.0f;
    // Byte distance between per-frame-slot copies of the indirect array
    // inside `indirect`. Non-zero lets the CPU rewrite the slot's region
    // (host-visible buffer) while the other slot's region is in flight —
    // the instanceCount 0/1 toggle path. 0 = one shared region.
    std::uint64_t indirectRegionStride = 0;
    // The camera matrix lives in the bindless table's per-slot camera
    // buffer (binding 6); only the slot index is pushed, so a moving
    // camera never invalidates static recordings.
    VkDescriptorSet descriptors = nullptr;  // bindless table set, bound once if set
    // Ray-traced primary visibility (optional): when rtPrimary is set and
    // the pipeline is present, the frame skips the shadow cascades, the
    // compaction dispatch and the indirect draw stream entirely and instead
    // draws one fullscreen triangle whose fragment shader traces the scene
    // through the TLAS (bindings 10-12). Toggling rtPrimary changes what
    // gets recorded — invalidate static recordings after flipping it.
    const Pipeline* rtPrimaryPipeline = nullptr;
    bool rtPrimary = false;
    // Keep the shadow cascades (and, in IndirectCount mode, the compaction
    // dispatch that feeds them) recorded even under rtPrimary: the traced
    // pass's volumetric fog march samples the cascade maps per step. Set
    // when the scene has fog; baked like rtPrimary — invalidate on change.
    bool fogCascades = false;
    // THE raster scene path (deferred shading; requires setDeferredTargets
    // called once): the opaque stream is rasterized into the G-buffer
    // targets (gbufferPipeline: scene VS + attribute-MRT fragment shader),
    // then the composite pass on the swapchain starts with one fullscreen
    // lighting triangle (lightingPipeline, depth test off) that reads
    // bindings 28-31 before the sky / transparent / proxy draws. Both
    // REQUIRED on every raster batch — there is no forward opaque path.
    const Pipeline* gbufferPipeline = nullptr;
    const Pipeline* lightingPipeline = nullptr;
    // Post-process pass (optional): when set (and the scene-color target
    // exists), every composite-pass draw — lighting, sky, transparents,
    // proxies, or the traced-primary triangle — renders into the RGBA16F
    // scene-color image (binding 35) instead of the swapchain, and one
    // final fullscreen triangle (this pipeline, colorFormat = swapchain,
    // no depth) maps it to the swapchain: exposure + tonemap, both riding
    // the per-slot light buffer so they stay live under static
    // recordings. Null = the composite pass targets the swapchain
    // directly (the pass pipelines must then declare the swapchain
    // format). Presence is baked — invalidate on change.
    const Pipeline* postPipeline = nullptr;
    // GPU skinning (optional): compute dispatches that pose animated
    // vertices into per-slot pool regions before any draw pass reads them.
    // push holds the shader's PushConstants with the slot element patched
    // at record time (kSkinSlotPushIndex).
    static constexpr std::size_t kSkinPushWords = 11;
    static constexpr std::size_t kSkinSlotPushIndex = 8;
    struct SkinDispatch {
        std::array<std::uint32_t, kSkinPushWords> push{};
        std::uint32_t vertexCount = 0;
    };
    const Pipeline* skinPipeline = nullptr;
    std::vector<SkinDispatch> skinDispatches;
    // Acceleration-structure upkeep before the passes. refitBlas (needs
    // skin dispatches): after the skin pass poses this slot's vertices,
    // the whole-scene BLAS is updated in place from refitGeometries[slot]
    // — the build-time geometry list with animated entries' vertex data
    // pointing at that slot's posed region (built with allowUpdate).
    // tlasRebuild: a DYNAMIC TLAS re-built in place every frame from the
    // slot's host-visible instance region (buildTopLevelDynamic), so
    // runtime-spawned instances and their transforms reach traced passes
    // without touching recordings or descriptors.
    const AccelerationStructure* refitBlas = nullptr;
    const AccelerationStructure* tlasRebuild = nullptr;
    std::vector<std::vector<AccelerationStructure::TriangleGeometry>> refitGeometries;
};

// Per-frame-recorded baseline frame loop: acquire, record, submit, present,
// with two frames in flight. Deliberately the naive re-record-every-frame
// path — the static command buffer model (docs/ARCHITECTURE.md) is measured
// against this in a later milestone.
class FrameRenderer {
public:
    static constexpr std::uint32_t kFramesInFlight = 2;

    static Result<std::unique_ptr<FrameRenderer>> create(const Device& device, Swapchain& swapchain);
    ~FrameRenderer();

    FrameRenderer(const FrameRenderer&) = delete;
    FrameRenderer& operator=(const FrameRenderer&) = delete;

    // Records and submits one frame. With a batch, the frame renders it
    // through the batch's own pipelines (G-buffer + lighting, or traced
    // primary); without one only the clear color (plus any overlay) is
    // presented — the app withholds the batch while a loading cover
    // hides the scene. Out-of-date/suboptimal swapchains are recreated
    // transparently.
    Result<void> drawFrame(const DrawBatch* batch = nullptr);

    // Static recording (the milestone-7 experiment): command buffers are
    // recorded once per (frame slot, swapchain image) and reused every
    // frame — per-frame CPU work shrinks to buffer writes + submit. The
    // recordings are invalidated (and lazily rebuilt on the next drawFrame)
    // by a swapchain recreate; visibility changes must go through the
    // indirect buffer, never through re-recording.
    void setStaticRecording(bool enabled);
    bool staticRecording() const { return staticEnabled_; }

    // Drops any static recordings (they rebuild lazily on the next
    // drawFrame). Call after changing what a frame records — e.g. flipping
    // DrawBatch::rtPrimary. Waits for the device to go idle first.
    void invalidateStaticRecordings() { invalidateStatic(); }

    // Per-frame overlay (UI) recorded into its own small command buffer
    // after the scene: the callback runs inside an active dynamic rendering
    // pass on the swapchain image (loadOp LOAD, viewport/scissor set) and
    // records e.g. ImGui draw data. Recorded every frame regardless of
    // static mode — overlay content is inherently dynamic; the prerecorded
    // scene buffers stay untouched. Null disables the pass.
    using OverlayRecorder = std::function<void(VkCommandBuffer)>;
    void setOverlayRecorder(OverlayRecorder recorder) { overlayRecorder_ = std::move(recorder); }

    // Blocks until the current frame slot's previous submission finished,
    // making the slot's per-frame regions (see DrawBatch::
    // indirectRegionStride) safe to write. drawFrame's own wait then
    // returns immediately.
    Result<void> waitFrameSlot();
    std::uint32_t frameSlot() const { return frameIndex_; }

    // CPU cost counters since the last take; the record/reset time is the
    // number the static-vs-rerecord experiment compares.
    struct Stats {
        std::uint64_t frames = 0;
        std::uint64_t recordMicros = 0; // per-frame reset+record CPU time
        std::uint64_t prerecords = 0;   // static recordings built (amortized)
    };
    Stats takeStats();

    // Requests a swapchain rebuild at the next frame; a zero size parks the
    // loop until a real size arrives (minimized window).
    void resize(std::uint32_t width, std::uint32_t height);

    // The destructor only touches device-owned objects, so the swapchain may
    // be destroyed first (it retires presents still waiting on the per-image
    // semaphores destroyed here).
    void waitIdle() const;

    void setClearColor(float r, float g, float b, float a = 1.0f) { clearColor_ = {r, g, b, a}; }

    // Deferred G-buffer targets: creates the four screen-sized images
    // (albedo / world normal / material / view depth) and writes them to
    // the table's bindings 28-31; both are redone on every swapchain
    // recreate (the device is idle there, and the bindings are not
    // update-after-bind). Call once at startup before the first frame.
    // Null disables deferred targets. The number/order/formats here must
    // match the gbuffer pipeline's colorFormats and the lighting shader.
    static constexpr std::uint32_t kGBufferTargets = 4;
    // Attachment order = descriptor binding order (28-31): albedo (sRGB),
    // world normal (16F), material params (unorm), view depth (32F, the
    // cleared 0 marks background). The gbuffer pipeline's colorFormats and
    // the lighting shader's Loads follow this order.
    static constexpr std::array<std::uint32_t, kGBufferTargets> kGBufferFormats{
        kFormatR8G8B8A8Srgb, kFormatR16G16B16A16Sfloat, kFormatR8G8B8A8Unorm, kFormatR32Sfloat};
    // The HDR scene-color target the composite pass renders into when a
    // batch carries a postPipeline (descriptor binding 35). Every pass
    // pipeline drawing inside the composite pass must declare this format.
    static constexpr std::uint32_t kSceneColorFormat = kFormatR16G16B16A16Sfloat;
    Result<void> setDeferredTargets(DescriptorTable* table);

private:
    FrameRenderer() = default;

    Result<void> createSyncObjects();
    Result<void> createImageSemaphores();
    void destroyImageSemaphores();
    Result<void> createDepthBuffer();
    Result<void> createGBuffer();
    Result<void> recreateSwapchain();
    Result<void> waitForFence(VkFence fence, const char* what) const;
    Result<void> record(VkCommandBuffer cmd, std::uint32_t imageIndex, std::uint32_t slot,
                        const DrawBatch* batch, bool reusable) const;
    Result<void> prerecordStatic(const DrawBatch* batch);
    void invalidateStatic();
    Result<void> recordOverlay(VkCommandBuffer cmd, std::uint32_t imageIndex) const;

    struct FrameData {
        VkCommandBuffer commandBuffer = nullptr;
        // The per-frame UI/present-transition tail after the scene buffer.
        VkCommandBuffer overlayCommandBuffer = nullptr;
        VkSemaphore imageAvailable = nullptr;
        VkFence inFlight = nullptr;
    };

    const Device* device_ = nullptr;
    Swapchain* swapchain_ = nullptr;
    VkCommandPool commandPool_ = nullptr;
    std::array<FrameData, kFramesInFlight> frames_{};
    // One per swapchain image, not per frame in flight: presentation may
    // still be reading an image's semaphore when its frame slot comes round.
    std::vector<VkSemaphore> renderFinished_;
    // Depth buffer at swapchain extent; recreated with it. One is enough
    // for both frames in flight: rendering is serialized by the barriers.
    std::unique_ptr<Image> depth_;
    // Deferred G-buffer targets at swapchain extent (albedo / normal /
    // material / view depth); recreated with it and rewritten into the
    // table's bindings 28-31. Single-instance like depth_ — rendering is
    // serialized by the barriers. Empty when deferred targets are off.
    std::array<std::unique_ptr<Image>, kGBufferTargets> gbuffer_{};
    // HDR scene-color target for the post pass (binding 35); lives and
    // recreates alongside the G-buffer.
    std::unique_ptr<Image> sceneColor_;
    DescriptorTable* deferredTable_ = nullptr;

    // Static-mode recordings, indexed [slot * imageCount + imageIndex];
    // empty while invalid. A (slot, image) pair is never in flight twice,
    // so the buffers need no simultaneous-use flag.
    std::vector<VkCommandBuffer> staticBuffers_;
    bool staticEnabled_ = false;
    bool staticValid_ = false;

    Stats stats_{};
    OverlayRecorder overlayRecorder_;

    std::array<float, 4> clearColor_{0.02f, 0.02f, 0.04f, 1.0f};
    std::uint32_t frameIndex_ = 0;
    std::uint32_t pendingWidth_ = 0;
    std::uint32_t pendingHeight_ = 0;
    bool resizeRequested_ = false;
};

} // namespace rend::gpu

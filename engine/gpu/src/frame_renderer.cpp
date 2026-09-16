#include "rend/gpu/frame_renderer.h"

#include "rend/gpu/buffer.h"

#include "rend/core/log.h"
#include "rend/core/profile.h"
#include "rend/gpu/command_context.h"
#include "rend/gpu/descriptor_table.h"
#include "rend/gpu/device.h"
#include "rend/gpu/image.h"
#include "rend/gpu/pipeline.h"
#include "rend/gpu/swapchain.h"

#include "vulkan/vulkan_types.h"

#include <algorithm>
#include <cstring>
#include <format>

namespace rend::gpu {

Result<std::unique_ptr<FrameRenderer>> FrameRenderer::create(const Device& device,
                                                             Swapchain& swapchain) {
    switch (device.api()) {
    case Api::Vulkan: return VulkanFrameRenderer::create(device, swapchain);
    case Api::D3D12: break;
    }
    return Error{std::format("{} backend: FrameRenderer not implemented", apiName(device.api()))};
}

Result<void> FrameRenderer::createDepthBuffer() {
    auto depthResult = Image::create(*device_, {
                                                   .width = swapchain_->width(),
                                                   .height = swapchain_->height(),
                                                   .format = kFormatD32Sfloat,
                                                   .usage = kImageUsageDepthAttachment,
                                                   .depth = true,
                                               });
    if (!depthResult) {
        return Error{std::format("Depth buffer: {}", depthResult.error().message)};
    }
    depth_ = std::move(depthResult).value();
    return {};
}

Result<void> FrameRenderer::createGBuffer() {
    for (std::uint32_t i = 0; i < kGBufferTargets; ++i) {
        auto result = Image::create(*device_, {
                                                  .width = swapchain_->width(),
                                                  .height = swapchain_->height(),
                                                  .format = kGBufferFormats[i],
                                                  .usage = kImageUsageColorAttachment |
                                                           kImageUsageSampled,
                                              });
        if (!result) {
            return Error{std::format("G-buffer target {}: {}", i, result.error().message)};
        }
        gbuffer_[i] = std::move(result).value();
        deferredTable_->writeSampledImage(28 + i, 0, *gbuffer_[i]);
    }
    // The HDR scene-color target the composite pass renders into when the
    // batch carries a post pipeline; the post pass Loads it (binding 35).
    auto sceneColorResult = Image::create(*device_, {
                                                        .width = swapchain_->width(),
                                                        .height = swapchain_->height(),
                                                        .format = kSceneColorFormat,
                                                        .usage = kImageUsageColorAttachment |
                                                                 kImageUsageSampled,
                                                    });
    if (!sceneColorResult) {
        return Error{std::format("Scene-color target: {}", sceneColorResult.error().message)};
    }
    sceneColor_ = std::move(sceneColorResult).value();
    deferredTable_->writeSampledImage(35, 0, *sceneColor_);
    // The LDR intermediate an AA module samples (post output parked one
    // pass before the swapchain when DrawBatch::aaPipeline is set).
    auto ldrResult = Image::create(*device_, {
                                                 .width = swapchain_->width(),
                                                 .height = swapchain_->height(),
                                                 .format = kLdrColorFormat,
                                                 .usage = kImageUsageColorAttachment |
                                                          kImageUsageSampled,
                                             });
    if (!ldrResult) {
        return Error{std::format("LDR color target: {}", ldrResult.error().message)};
    }
    ldrColor_ = std::move(ldrResult).value();
    deferredTable_->writeSampledImage(36, 0, *ldrColor_);
    return {};
}

Result<void> FrameRenderer::setDeferredTargets(DescriptorTable* table) {
    deferredTable_ = table;
    if (table == nullptr) {
        gbuffer_ = {};
        sceneColor_.reset();
        ldrColor_.reset();
        return {};
    }
    return createGBuffer();
}

void FrameRenderer::resize(std::uint32_t width, std::uint32_t height) {
    pendingWidth_ = width;
    pendingHeight_ = height;
    resizeRequested_ = true;
}

Result<void> FrameRenderer::recreateSwapchain() {
    const std::uint32_t width = pendingWidth_ != 0 ? pendingWidth_ : swapchain_->width();
    const std::uint32_t height = pendingHeight_ != 0 ? pendingHeight_ : swapchain_->height();
    resizeRequested_ = false;

    const std::uint32_t previousImageCount = swapchain_->imageCount();
    if (auto r = swapchain_->recreate(width, height); !r) {
        return r.error();
    }
    // Static recordings bake image handles and extent; rebuild lazily.
    invalidateStatic();
    // Per-image backend state (Vulkan: the render-finished semaphores)
    // follows the image count. The device is idle after recreate().
    if (auto r = onSwapchainRecreated(previousImageCount); !r) {
        return r.error();
    }
    // Depth tracks the swapchain extent (device idle here, see above).
    if (auto r = createDepthBuffer(); !r) {
        return r.error();
    }
    // The G-buffer tracks it too; the device-idle window also makes the
    // non-update-after-bind descriptor rewrites (bindings 28-31) safe.
    if (deferredTable_ != nullptr) {
        if (auto r = createGBuffer(); !r) {
            return r.error();
        }
    }
    return {};
}

// The whole scene frame, expressed through the neutral CommandContext so a
// second backend records the same passes from the same code. Every barrier
// below is the precise stage/access/layout scope the Vulkan version carried
// — do not "simplify" one without re-running the sync-validation benches.
void FrameRenderer::recordFrame(CommandContext& ctx, std::uint32_t imageIndex,
                                std::uint32_t slot, const DrawBatch* batch) const {
    // The bindless table every batch pass binds (null = none).
    const DescriptorTable* table = batch ? batch->descriptors : nullptr;
    auto bindTable = [&](const Pipeline& p) {
        if (table != nullptr) {
            ctx.bindDescriptorTable(p, *table);
        }
    };
    // Fullscreen-triangle passes push {slot, 0} (the layout the scene
    // passes share).
    auto pushSlot = [&](const Pipeline& p) {
        const std::uint32_t push[2] = {slot, 0};
        ctx.pushConstants(p, push, sizeof(push));
    };

    const Image& swapchainImage = swapchain_->image(imageIndex);

    // Frame passes see the frame through this context; the per-point
    // fields (color attachment, depth) are filled in as the frame goes.
    PassContext passContext{};
    passContext.slot = slot;
    passContext.imageIndex = imageIndex;
    passContext.width = swapchain_->width();
    passContext.height = swapchain_->height();
    passContext.swapchainImage = &swapchainImage;
    passContext.swapchainFormat = swapchain_->imageFormat();
    recordPasses(ctx, PassPoint::BeforeScene, passContext);

    // Traced primary visibility replaces the raster pipeline: no shadow
    // cascades, no compaction, no indirect stream — one fullscreen triangle
    // whose fragments walk the TLAS instead.
    const bool rtDraw = batch && batch->rtPrimary && batch->rtPrimaryPipeline;
    // Exception: a fog scene's traced pass marches the cascade maps, so
    // the compaction dispatch + cascade passes stay recorded for it.
    const bool fogCascades = rtDraw && batch->fogCascades;
    // The raster scene path IS deferred: G-buffer pass then the lighting
    // triangle in the composite pass. A batch must carry gbufferPipeline +
    // lightingPipeline (see DrawBatch) — there is no forward opaque path.
    const bool rasterScene = batch && !rtDraw;

    if (batch && batch->skinPipeline && !batch->skinDispatches.empty()) {
        // Pose animated vertices into this slot's pool regions. Order
        // against the previous frame's vertex fetches, then make the
        // writes visible to every consumer of the pool this frame.
        // COMPUTE in the source: the previous frame's OBB refine dispatch
        // reads the pool too (WAR — execution ordering is enough).
        ctx.memoryBarrier({.srcStage = PipelineStage::VertexAttributeInput |
                                       PipelineStage::IndexInput | PipelineStage::ComputeShader,
                           .srcAccess = Access::None,
                           .dstStage = PipelineStage::ComputeShader,
                           .dstAccess = Access::ShaderStorageRead | Access::ShaderStorageWrite});

        ctx.bindPipeline(*batch->skinPipeline);
        bindTable(*batch->skinPipeline);
        for (const DrawBatch::SkinDispatch& dispatch : batch->skinDispatches) {
            auto push = dispatch.push;
            push[DrawBatch::kSkinSlotPushIndex] = slot;
            ctx.pushConstants(*batch->skinPipeline, push.data(),
                              static_cast<std::uint32_t>(sizeof(push)));
            ctx.dispatch((dispatch.vertexCount + 63) / 64, 1, 1);
        }

        // COMPUTE in the destination: this frame's OBB refine dispatch
        // reads the pool (disjoint regions from the posed writes, but the
        // hazard tracking is buffer-wide).
        ctx.memoryBarrier({.srcStage = PipelineStage::ComputeShader,
                           .srcAccess = Access::ShaderStorageWrite,
                           .dstStage = PipelineStage::VertexAttributeInput |
                                       PipelineStage::FragmentShader | PipelineStage::ComputeShader,
                           .dstAccess = Access::VertexAttributeRead | Access::ShaderStorageRead});
    }

    const bool blasRefit = batch && batch->refitBlas && slot < batch->refitGeometries.size() &&
                           !batch->refitGeometries[slot].empty();
    if (blasRefit || (batch && batch->tlasRebuild)) {
        // Refit the BLAS from this slot's posed vertices (animated scenes)
        // and/or re-build the dynamic TLAS from this slot's instance
        // region (runtime models + their transforms). The entry barrier
        // makes the skin writes visible to the build AND — queue-scoped,
        // so it reaches across submissions — orders it after the previous
        // frame in flight's ray queries reading the same structures.
        ctx.memoryBarrier({.srcStage = PipelineStage::ComputeShader | PipelineStage::FragmentShader,
                           .srcAccess = Access::ShaderStorageWrite,
                           .dstStage = PipelineStage::AccelerationStructureBuild,
                           .dstAccess = Access::ShaderRead | Access::AccelerationStructureRead |
                                        Access::AccelerationStructureWrite});

        if (blasRefit) {
            batch->refitBlas->recordRefit(ctx, batch->refitGeometries[slot]);
            // The TLAS rebuild reads the refitted BLAS AABBs.
            ctx.memoryBarrier({.srcStage = PipelineStage::AccelerationStructureBuild,
                               .srcAccess = Access::AccelerationStructureWrite,
                               .dstStage = PipelineStage::AccelerationStructureBuild,
                               .dstAccess = Access::AccelerationStructureRead |
                                            Access::AccelerationStructureWrite});
        }

        if (batch->tlasRebuild) {
            batch->tlasRebuild->recordRebuild(ctx, slot);
        }

        ctx.memoryBarrier({.srcStage = PipelineStage::AccelerationStructureBuild,
                           .srcAccess = Access::AccelerationStructureWrite,
                           .dstStage = PipelineStage::FragmentShader,
                           .dstAccess = Access::AccelerationStructureRead});
    }

    if (batch && (!rtDraw || fogCascades) && batch->cullPipeline &&
        batch->mode == DrawSubmitMode::IndirectCount) {
        // GPU compaction: zero the slot's whole counter region (stream
        // counts, scratch-row allocator, emitted-index stats), run one
        // thread per template, then make the writes visible to the
        // indirect fetch.
        // Under traced primaries the proxy pass never runs, so occlusion
        // testing here would read (and keep re-clearing) a visibility
        // buffer nothing refreshes — every entry would drop, and the
        // first raster frame after a mode toggle would flash blank.
        // Mask the bit out: visibility stays stale-but-conservative,
        // exactly like scenes where this dispatch never ran.
        const std::uint32_t cullFlags = rtDraw ? (batch->cullFlags & ~4u) : batch->cullFlags;
        const bool occlusion = batch->occlusionPipeline != nullptr &&
                               batch->occlusionVisibility != nullptr && (cullFlags & 4u) != 0;
        // The visibility clear happens AFTER the cull dispatch below:
        // with two-slot hysteresis the dispatch reads BOTH regions (this
        // slot's still holds the frame-before-last's proxy results), so
        // this slot's region must survive until then before the proxy
        // pass at the end of this frame refills it.
        ctx.fillBuffer(*batch->count, slot * batch->countRegionStride, batch->countRegionStride,
                       0);

        // The fills must land before the dispatch reads/increments, and —
        // with occlusion — the previous frame's proxy-pass fragment
        // stores into the other slot's region must be visible too. With
        // OBB refinement the previous frame's refine writes (COMPUTE) and
        // proxy-VS OBB reads (VERTEX, WAR) join the source scope: this
        // frame's refine dispatch re-reads the counter and overwrites
        // rows the previous frame's consumers looked at.
        const bool obbRefine = batch->obbRefinePipeline != nullptr;
        PipelineStage fillSrc = PipelineStage::Clear;
        if (occlusion) {
            fillSrc = fillSrc | PipelineStage::FragmentShader;
        }
        if (obbRefine) {
            fillSrc = fillSrc | PipelineStage::ComputeShader | PipelineStage::VertexShader;
        }
        Access fillSrcAccess = Access::TransferWrite;
        if (occlusion || obbRefine) {
            fillSrcAccess = fillSrcAccess | Access::ShaderStorageWrite;
        }
        ctx.memoryBarrier({.srcStage = fillSrc,
                           .srcAccess = fillSrcAccess,
                           .dstStage = PipelineStage::ComputeShader,
                           .dstAccess = Access::ShaderStorageRead | Access::ShaderStorageWrite});

        if (obbRefine) {
            // Self-terminating OBB refinement: claims the next
            // obbRefineGroups entries from binding 33's counter and fits
            // their oriented boxes; exits immediately once every entry
            // has been claimed. Baked like everything here — the counter
            // is the only state, so no recording ever changes.
            ctx.bindPipeline(*batch->obbRefinePipeline);
            bindTable(*batch->obbRefinePipeline);
            const std::uint32_t obbPush[3] = {
                batch->drawCount, slot,
                static_cast<std::uint32_t>(batch->indirectRegionStride /
                                           sizeof(DrawIndexedIndirect))};
            ctx.pushConstants(*batch->obbRefinePipeline, obbPush, sizeof(obbPush));
            ctx.dispatch(batch->obbRefineGroups, 1, 1);

            // The cull dispatch reads the rows the refine pass just wrote.
            ctx.memoryBarrier({.srcStage = PipelineStage::ComputeShader,
                               .srcAccess = Access::ShaderStorageWrite,
                               .dstStage = PipelineStage::ComputeShader,
                               .dstAccess = Access::ShaderStorageRead});
        }

        ctx.bindPipeline(*batch->cullPipeline);
        bindTable(*batch->cullPipeline);
        // capacity = the per-slot region stride in entries; drawCount can
        // grow at runtime (model loads) while the regions stay put. The
        // fifth word is the LOD screen-size factor, a float in disguise.
        std::uint32_t push[5] = {
            batch->drawCount, slot,
            static_cast<std::uint32_t>(batch->indirectRegionStride / sizeof(DrawIndexedIndirect)),
            cullFlags, 0};
        std::memcpy(&push[4], &batch->lodFactor, sizeof(float));
        ctx.pushConstants(*batch->cullPipeline, push, sizeof(push));
        ctx.dispatch((batch->drawCount + 63) / 64, 1, 1);

        // The vertex stage also reads what the cull pass wrote: partially
        // visible draws' surviving instance rows land in the rows buffer's
        // per-slot scratch regions.
        ctx.memoryBarrier({.srcStage = PipelineStage::ComputeShader,
                           .srcAccess = Access::ShaderStorageWrite,
                           .dstStage = PipelineStage::DrawIndirect | PipelineStage::VertexShader,
                           .dstAccess = Access::IndirectCommandRead | Access::ShaderStorageRead});

        if (occlusion) {
            // Now that the cull dispatch has read this slot's region
            // (hysteresis), zero it for the proxy pass at the end of THIS
            // frame. WAR against the dispatch, then make the clear
            // visible to the proxy fragments' visibility stores.
            ctx.memoryBarrier({.srcStage = PipelineStage::ComputeShader,
                               .srcAccess = Access::ShaderStorageRead,
                               .dstStage = PipelineStage::Clear,
                               .dstAccess = Access::TransferWrite});
            ctx.fillBuffer(*batch->occlusionVisibility, slot * batch->occlusionRegionStride,
                           batch->occlusionRegionStride, 0);
            ctx.memoryBarrier({.srcStage = PipelineStage::Clear,
                               .srcAccess = Access::TransferWrite,
                               .dstStage = PipelineStage::FragmentShader,
                               .dstAccess = Access::ShaderStorageRead | Access::ShaderStorageWrite});
        }
    }

    // Binds the batch's geometry/descriptors and emits its draw stream
    // with the given pipeline — shared by the shadow, main and blend
    // passes. The cascade index only matters to the shadow pipeline.
    // stream: 0 = the visibility-only list the shadow passes draw, 1 = the
    // frustum-culled opaque list (when the batch carries one), 2 = the
    // frustum-culled transparent list.
    auto bindAndDraw = [&](const Pipeline& p, std::uint32_t cascade, std::uint32_t stream) {
        bindTable(p);
        ctx.bindVertexBuffer(*batch->geometry, 0);
        ctx.bindIndexBuffer(*batch->geometry, 0);
        const std::uint32_t push[2] = {slot, cascade};
        ctx.pushConstants(p, push, sizeof(push));
        // Counter layout per slot: [0] shadow, [1] opaque, [2] scratch
        // rows, [3] transparent.
        const Buffer* streamBuffer = batch->indirect;
        std::uint64_t countOffset = 0;
        if (stream == 1 && batch->sceneIndirect != nullptr) {
            streamBuffer = batch->sceneIndirect;
            countOffset = 1 * sizeof(std::uint32_t);
        } else if (stream == 2) {
            streamBuffer = batch->transparentIndirect;
            countOffset = 3 * sizeof(std::uint32_t);
        }
        switch (batch->mode) {
        case DrawSubmitMode::IndirectCount:
            ctx.drawIndexedIndirectCount(*streamBuffer, slot * batch->indirectRegionStride,
                                         *batch->count,
                                         slot * batch->countRegionStride + countOffset,
                                         batch->drawCount, sizeof(DrawIndexedIndirect));
            break;
        case DrawSubmitMode::Indirect:
            ctx.drawIndexedIndirect(*batch->indirect, slot * batch->indirectRegionStride,
                                    batch->drawCount, sizeof(DrawIndexedIndirect));
            break;
        case DrawSubmitMode::Direct:
            for (std::uint32_t i = 0; i < batch->drawCount; ++i) {
                const DrawIndexedIndirect& draw = batch->cpuDraws[i];
                if (draw.instanceCount != 0) {
                    ctx.drawIndexed(draw.indexCount, draw.instanceCount, draw.firstIndex,
                                    draw.vertexOffset, draw.firstInstance);
                }
            }
            break;
        }
    };

    if (batch && (!rtDraw || fogCascades) && batch->shadowPipeline && batch->cascadeCount > 0) {
        for (std::uint32_t c = 0; c < batch->cascadeCount; ++c) {
            const Image* map = batch->shadowCascades[c];
            // Depth-only pass from the light's view. Contents are cleared,
            // so the old layout is UNDEFINED; the barrier orders against
            // the previous frame's sampling and depth writes.
            ctx.imageBarrier({.image = map,
                              .oldLayout = ImageLayout::Undefined,
                              .newLayout = ImageLayout::DepthAttachment,
                              .srcStage = PipelineStage::FragmentShader |
                                          PipelineStage::LateFragmentTests,
                              .srcAccess = Access::DepthStencilWrite,
                              .dstStage = PipelineStage::EarlyFragmentTests |
                                          PipelineStage::LateFragmentTests,
                              .dstAccess = Access::DepthStencilRead | Access::DepthStencilWrite});

            const DepthTarget shadowDepth{
                .image = map, .load = LoadOp::Clear, .store = true, .clear = 1.0f};
            RenderingDesc shadowRendering{};
            shadowRendering.width = map->width();
            shadowRendering.height = map->height();
            shadowRendering.depth = &shadowDepth;
            ctx.beginRendering(shadowRendering);
            ctx.bindPipeline(*batch->shadowPipeline);
            bindAndDraw(*batch->shadowPipeline, c, 0);
            ctx.endRendering();

            // Written depth becomes sampleable by the main pass's fragments.
            ctx.imageBarrier({.image = map,
                              .oldLayout = ImageLayout::DepthAttachment,
                              .newLayout = ImageLayout::ShaderReadOnly,
                              .srcStage = PipelineStage::LateFragmentTests,
                              .srcAccess = Access::DepthStencilWrite,
                              .dstStage = PipelineStage::FragmentShader,
                              .dstAccess = Access::ShaderSampledRead});
        }
    }

    if (rasterScene) {
        // G-buffer pass: rasterize the opaque stream's surface attributes
        // into the four screen-sized targets plus depth. Target contents
        // are cleared, so old layouts are UNDEFINED; the barriers order
        // against the previous frame's lighting-pass reads. Depth is
        // cleared here and consumed by the composite pass's depth-tested
        // draws (sky/transparent/proxy); its barrier orders against the
        // previous frame's depth accesses.
        std::array<ImageBarrierDesc, kGBufferTargets + 1> toGBufferWrite{};
        for (std::uint32_t i = 0; i < kGBufferTargets; ++i) {
            toGBufferWrite[i] = {.image = gbuffer_[i].get(),
                                 .oldLayout = ImageLayout::Undefined,
                                 .newLayout = ImageLayout::ColorAttachment,
                                 .srcStage = PipelineStage::FragmentShader,
                                 .srcAccess = Access::None,
                                 .dstStage = PipelineStage::ColorAttachmentOutput,
                                 .dstAccess = Access::ColorAttachmentWrite};
        }
        toGBufferWrite.back() = {
            .image = depth_.get(),
            .oldLayout = ImageLayout::Undefined,
            .newLayout = ImageLayout::DepthAttachment,
            .srcStage = PipelineStage::EarlyFragmentTests | PipelineStage::LateFragmentTests,
            .srcAccess = Access::DepthStencilWrite,
            .dstStage = PipelineStage::EarlyFragmentTests | PipelineStage::LateFragmentTests,
            .dstAccess = Access::DepthStencilRead | Access::DepthStencilWrite};
        ctx.barrier({}, toGBufferWrite);

        RenderingDesc gbufferRendering{};
        gbufferRendering.width = swapchain_->width();
        gbufferRendering.height = swapchain_->height();
        for (std::uint32_t i = 0; i < kGBufferTargets; ++i) {
            // All-zero clears: the zeroed view-depth target is the
            // lighting pass's background sentinel.
            gbufferRendering.colors.push_back(ColorTarget{.image = gbuffer_[i].get(),
                                                          .load = LoadOp::Clear,
                                                          .store = true,
                                                          .clear = {0.0f, 0.0f, 0.0f, 0.0f}});
        }
        const DepthTarget gbufferDepth{.image = depth_.get(),
                                       .load = LoadOp::Clear,
                                       .store = true, // composite pass loads it
                                       .clear = 1.0f};
        gbufferRendering.depth = &gbufferDepth;
        ctx.beginRendering(gbufferRendering);
        ctx.bindPipeline(*batch->gbufferPipeline);
        bindAndDraw(*batch->gbufferPipeline, 0, 1);
        ctx.endRendering();

        // Written targets become sampleable by the lighting triangle, and
        // the stored depth becomes testable by the composite pass's
        // depth-tested draws (sky/transparent/proxy).
        std::array<ImageBarrierDesc, kGBufferTargets + 1> toRead{};
        for (std::uint32_t i = 0; i < kGBufferTargets; ++i) {
            toRead[i] = {.image = gbuffer_[i].get(),
                         .oldLayout = ImageLayout::ColorAttachment,
                         .newLayout = ImageLayout::ShaderReadOnly,
                         .srcStage = PipelineStage::ColorAttachmentOutput,
                         .srcAccess = Access::ColorAttachmentWrite,
                         .dstStage = PipelineStage::FragmentShader,
                         .dstAccess = Access::ShaderSampledRead};
        }
        toRead.back() = {
            .image = depth_.get(),
            .oldLayout = ImageLayout::DepthAttachment,
            .newLayout = ImageLayout::DepthAttachment,
            .srcStage = PipelineStage::LateFragmentTests,
            .srcAccess = Access::DepthStencilWrite,
            .dstStage = PipelineStage::EarlyFragmentTests | PipelineStage::LateFragmentTests,
            .dstAccess = Access::DepthStencilRead};
        ctx.barrier({}, toRead);
    }

    // Post mode: the composite pass renders into the HDR scene-color
    // target and a final fullscreen pass (exposure + tonemap) maps it to
    // the swapchain. Every composite-pass pipeline must then declare
    // kSceneColorFormat. Without a post pipeline the composite pass
    // targets the swapchain directly, exactly as before.
    const bool post = batch && batch->postPipeline != nullptr && sceneColor_ != nullptr;

    if (post) {
        // Scene color is cleared each frame (old layout UNDEFINED); order
        // against the previous frame's post-pass sampled reads.
        ctx.imageBarrier({.image = sceneColor_.get(),
                          .oldLayout = ImageLayout::Undefined,
                          .newLayout = ImageLayout::ColorAttachment,
                          .srcStage = PipelineStage::FragmentShader,
                          .srcAccess = Access::None,
                          .dstStage = PipelineStage::ColorAttachmentOutput,
                          .dstAccess = Access::ColorAttachmentWrite});
    } else {
        ctx.imageBarrier({.image = &swapchainImage,
                          .oldLayout = ImageLayout::Undefined,
                          .newLayout = ImageLayout::ColorAttachment,
                          .srcStage = PipelineStage::ColorAttachmentOutput,
                          .srcAccess = Access::None,
                          .dstStage = PipelineStage::ColorAttachmentOutput,
                          .dstAccess = Access::ColorAttachmentWrite});
    }

    const Image* compositeTarget = post ? sceneColor_.get() : &swapchainImage;
    RenderingDesc composite{};
    composite.width = swapchain_->width();
    composite.height = swapchain_->height();
    composite.colors.push_back(ColorTarget{.image = compositeTarget,
                                           .load = LoadOp::Clear,
                                           .store = true,
                                           .clear = clearColor_});
    // Loads the G-buffer pass's stored depth — the sky/transparent/proxy
    // draws test against it; nothing here writes it. Depth only when
    // rasterizing a batch: the attachment set must match the pipeline's
    // declared depthFormat (traced primary needs none).
    const DepthTarget compositeDepth{
        .image = depth_.get(), .load = LoadOp::Load, .store = false, .clear = 1.0f};
    composite.depth = rasterScene ? &compositeDepth : nullptr;
    ctx.beginRendering(composite);

    if (rtDraw) {
        // Fullscreen traced pass: every pixel fires a camera ray in the
        // fragment shader; camera/light/slot data flow exactly as in the
        // raster path, so static recordings survive camera motion here too.
        ctx.bindPipeline(*batch->rtPrimaryPipeline);
        bindTable(*batch->rtPrimaryPipeline);
        pushSlot(*batch->rtPrimaryPipeline);
        ctx.draw(3, 1, 0, 0);
    } else if (batch) {
        // Lighting: one fullscreen triangle Loads the G-buffer (bindings
        // 28-31) and shades every covered pixel; the opaque stream was
        // already rasterized in the G-buffer pass. Background pixels
        // discard, keeping the baked clear color for the sky pass to
        // overdraw.
        ctx.bindPipeline(*batch->lightingPipeline);
        bindTable(*batch->lightingPipeline);
        pushSlot(*batch->lightingPipeline);
        ctx.draw(3, 1, 0, 0);
        // Sky pass: fullscreen triangle at the far plane, depth test
        // only — paints the per-slot skyColor over background pixels
        // before the transparents blend on top of it.
        if (batch->skyPipeline != nullptr) {
            ctx.bindPipeline(*batch->skyPipeline);
            bindTable(*batch->skyPipeline);
            pushSlot(*batch->skyPipeline);
            ctx.draw(3, 1, 0, 0);
        }
        // Transparency pass: same rendering pass, blend pipeline,
        // depth write off — the cull shader routed these entries out
        // of the opaque stream. Unsorted for now (single-layer glass
        // is fine; stacked transparents may blend out of order).
        if (batch->transparentPipeline && batch->transparentIndirect != nullptr &&
            batch->mode == DrawSubmitMode::IndirectCount) {
            ctx.bindPipeline(*batch->transparentPipeline);
            bindAndDraw(*batch->transparentPipeline, 0, 2);
        }
        // The proxy and debug passes push {slot, capacity}.
        const std::uint32_t proxyPush[2] = {
            slot,
            static_cast<std::uint32_t>(batch->indirectRegionStride / sizeof(DrawIndexedIndirect))};
        // Occlusion proxy pass: every template's world AABB as an
        // instanced cube against the frame's finished depth (test
        // only, color masked); survivors mark the visibility buffer
        // the NEXT frame's cull dispatch consumes. Same rendering
        // pass, so the scene's depth writes are already ordered.
        if (batch->occlusionPipeline != nullptr && (batch->cullFlags & 4u) != 0 &&
            batch->mode == DrawSubmitMode::IndirectCount) {
            ctx.bindPipeline(*batch->occlusionPipeline);
            bindTable(*batch->occlusionPipeline);
            ctx.pushConstants(*batch->occlusionPipeline, proxyPush, sizeof(proxyPush));
            ctx.draw(36, batch->drawCount, 0, 0);
            // Per-instance proxy pass: one box per canonical instance
            // row (dead/scene rows emit degenerate geometry). Same
            // push constants and PS; marks the region's per-instance
            // visibility slots.
            if (batch->occlusionInstancePipeline != nullptr && batch->occlusionInstanceRows > 0) {
                ctx.bindPipeline(*batch->occlusionInstancePipeline);
                bindTable(*batch->occlusionInstancePipeline);
                ctx.pushConstants(*batch->occlusionInstancePipeline, proxyPush,
                                  sizeof(proxyPush));
                ctx.draw(36, batch->occlusionInstanceRows, 0, 0);
            }
        }
        // Occlusion-box debug overlay: the same instanced AABB cubes
        // as translucent color (identical depth state, so the tinted
        // fragments are exactly the proxy pass's survivors). Not
        // gated on cullFlags bit 2 — the boxes are inspectable with
        // occlusion culling toggled off.
        if (batch->occlusionDebugPipeline != nullptr &&
            batch->mode == DrawSubmitMode::IndirectCount) {
            ctx.bindPipeline(*batch->occlusionDebugPipeline);
            bindTable(*batch->occlusionDebugPipeline);
            ctx.pushConstants(*batch->occlusionDebugPipeline, proxyPush, sizeof(proxyPush));
            ctx.draw(36, batch->drawCount, 0, 0);
            // The instanced models' boxes, same overlay styling.
            if (batch->occlusionInstanceDebugPipeline != nullptr &&
                batch->occlusionInstanceRows > 0) {
                ctx.bindPipeline(*batch->occlusionInstanceDebugPipeline);
                bindTable(*batch->occlusionInstanceDebugPipeline);
                ctx.pushConstants(*batch->occlusionInstanceDebugPipeline, proxyPush,
                                  sizeof(proxyPush));
                ctx.draw(36, batch->occlusionInstanceRows, 0, 0);
            }
        }
    }
    // No batch: nothing draws — the cleared swapchain image (plus the
    // overlay pass) is the whole frame — unless InScene passes draw here.
    passContext.color = compositeTarget;
    passContext.colorFormat = post ? kSceneColorFormat : swapchain_->imageFormat();
    passContext.depthAttached = rasterScene;
    passContext.depth = rasterScene ? depth_.get() : nullptr;
    recordPasses(ctx, PassPoint::InScene, passContext);

    ctx.endRendering();

    passContext.color = nullptr;
    passContext.colorFormat = Format::Undefined;
    passContext.depthAttached = false;
    passContext.depth = nullptr;
    recordPasses(ctx, PassPoint::AfterScene, passContext);

    if (post) {
        // With an AA module the post pass parks its tonemapped output in
        // the LDR intermediate and the AA pass maps that to the
        // swapchain; without one the post pass writes the swapchain
        // directly.
        const bool aa = batch->aaPipeline != nullptr && batch->postLdrPipeline != nullptr &&
                        ldrColor_ != nullptr;

        // Shared fullscreen-pass recorder: bind, push {slot, 0}, draw one
        // triangle into the target (which covers every pixel, so the old
        // contents are never loaded).
        auto fullscreenPass = [&](const Pipeline& p, const Image& target) {
            RenderingDesc pass{};
            pass.width = swapchain_->width();
            pass.height = swapchain_->height();
            pass.colors.push_back(
                ColorTarget{.image = &target, .load = LoadOp::DontCare, .store = true});
            ctx.beginRendering(pass);
            ctx.bindPipeline(p);
            bindTable(p);
            pushSlot(p);
            ctx.draw(3, 1, 0, 0);
            ctx.endRendering();
        };

        // Scene color becomes sampleable and the post target (LDR
        // intermediate or the swapchain) becomes an attachment. For the
        // LDR image the unordered prior access is last frame's AA
        // sampling.
        const std::array<ImageBarrierDesc, 2> toPost{
            ImageBarrierDesc{.image = sceneColor_.get(),
                             .oldLayout = ImageLayout::ColorAttachment,
                             .newLayout = ImageLayout::ShaderReadOnly,
                             .srcStage = PipelineStage::ColorAttachmentOutput,
                             .srcAccess = Access::ColorAttachmentWrite,
                             .dstStage = PipelineStage::FragmentShader,
                             .dstAccess = Access::ShaderSampledRead},
            ImageBarrierDesc{.image = aa ? ldrColor_.get() : &swapchainImage,
                             .oldLayout = ImageLayout::Undefined,
                             .newLayout = ImageLayout::ColorAttachment,
                             .srcStage = aa ? PipelineStage::FragmentShader
                                            : PipelineStage::ColorAttachmentOutput,
                             .srcAccess = Access::None,
                             .dstStage = PipelineStage::ColorAttachmentOutput,
                             .dstAccess = Access::ColorAttachmentWrite}};
        ctx.barrier({}, toPost);

        fullscreenPass(aa ? *batch->postLdrPipeline : *batch->postPipeline,
                       aa ? *ldrColor_ : swapchainImage);

        if (aa) {
            // LDR output becomes sampleable, the swapchain becomes the
            // attachment, and the AA module resolves onto it.
            const std::array<ImageBarrierDesc, 2> toAa{
                ImageBarrierDesc{.image = ldrColor_.get(),
                                 .oldLayout = ImageLayout::ColorAttachment,
                                 .newLayout = ImageLayout::ShaderReadOnly,
                                 .srcStage = PipelineStage::ColorAttachmentOutput,
                                 .srcAccess = Access::ColorAttachmentWrite,
                                 .dstStage = PipelineStage::FragmentShader,
                                 .dstAccess = Access::ShaderSampledRead},
                ImageBarrierDesc{.image = &swapchainImage,
                                 .oldLayout = ImageLayout::Undefined,
                                 .newLayout = ImageLayout::ColorAttachment,
                                 .srcStage = PipelineStage::ColorAttachmentOutput,
                                 .srcAccess = Access::None,
                                 .dstStage = PipelineStage::ColorAttachmentOutput,
                                 .dstAccess = Access::ColorAttachmentWrite}};
            ctx.barrier({}, toAa);

            fullscreenPass(*batch->aaPipeline, swapchainImage);
        }
    }

    recordPasses(ctx, PassPoint::AfterPost, passContext);
}

void FrameRenderer::recordPasses(CommandContext& ctx, PassPoint point,
                                 PassContext& context) const {
    context.point = point;
    for (const FramePass& pass : framePasses_) {
        if (pass.point == point && pass.record) {
            pass.record(ctx, context);
        }
    }
}

void FrameRenderer::setFramePasses(std::vector<FramePass> passes) {
    framePasses_ = std::move(passes);
    // Presence and content of the passes are baked into the recordings.
    invalidateStatic();
}

void FrameRenderer::recordOverlayFrame(CommandContext& ctx, std::uint32_t imageIndex) const {
    const Image& image = swapchain_->image(imageIndex);

    if (overlayRecorder_) {
        // Order against the scene buffer's color writes (same submission,
        // no layout change) before loading the attachment.
        ctx.imageBarrier({.image = &image,
                          .oldLayout = ImageLayout::ColorAttachment,
                          .newLayout = ImageLayout::ColorAttachment,
                          .srcStage = PipelineStage::ColorAttachmentOutput,
                          .srcAccess = Access::ColorAttachmentWrite,
                          .dstStage = PipelineStage::ColorAttachmentOutput,
                          .dstAccess = Access::ColorAttachmentRead | Access::ColorAttachmentWrite});

        RenderingDesc overlay{};
        overlay.width = swapchain_->width();
        overlay.height = swapchain_->height();
        overlay.colors.push_back(ColorTarget{.image = &image, .load = LoadOp::Load, .store = true});
        ctx.beginRendering(overlay);
        overlayRecorder_(ctx);
        ctx.endRendering();
    }

    ctx.imageBarrier({.image = &image,
                      .oldLayout = ImageLayout::ColorAttachment,
                      .newLayout = ImageLayout::Present,
                      .srcStage = PipelineStage::ColorAttachmentOutput,
                      .srcAccess = Access::ColorAttachmentWrite,
                      .dstStage = PipelineStage::AllCommands,
                      .dstAccess = Access::None});
}

void FrameRenderer::setStaticRecording(bool enabled) {
    if (staticEnabled_ == enabled) {
        return;
    }
    staticEnabled_ = enabled;
    invalidateStatic();
}

FrameRenderer::Stats FrameRenderer::takeStats() {
    Stats out = stats_;
    stats_ = {};
    return out;
}

} // namespace rend::gpu

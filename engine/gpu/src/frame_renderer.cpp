#include "rend/gpu/frame_renderer.h"

#include "rend/core/log.h"
#include "rend/core/profile.h"
#include "rend/gpu/descriptor_table.h"
#include "rend/gpu/device.h"
#include "rend/gpu/image.h"
#include "rend/gpu/pipeline.h"
#include "rend/gpu/swapchain.h"

#include <volk.h>

#include <algorithm>
#include <chrono>
#include <cstring>

namespace rend::gpu {

namespace {

// The frame loop never blocks forever: a fence or acquire that stalls this
// long is reported and retried, so a wedged driver stays diagnosable (and
// the process stays killable) instead of silently hanging.
constexpr std::uint64_t kWaitTimeoutNs = 2'000'000'000ull;
constexpr int kMaxStalledWaits = 5;

// Barrier helper: the swapchain image's previous contents are always
// discarded (loadOp CLEAR), so the source layout is UNDEFINED every frame.
VkImageMemoryBarrier2 imageBarrier(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                                   VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                                   VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return barrier;
}

} // namespace

Result<std::unique_ptr<FrameRenderer>> FrameRenderer::create(const Device& device,
                                                             Swapchain& swapchain) {
    auto renderer = std::unique_ptr<FrameRenderer>(new FrameRenderer());
    renderer->device_ = &device;
    renderer->swapchain_ = &swapchain;

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = device.graphicsQueue().familyIndex;
    if (VkResult r = vkCreateCommandPool(device.handle(), &poolInfo, nullptr, &renderer->commandPool_);
        r != VK_SUCCESS) {
        return Error{std::format("vkCreateCommandPool failed ({})", static_cast<int>(r))};
    }

    if (auto r = renderer->createSyncObjects(); !r) {
        return r.error();
    }
    if (auto r = renderer->createImageSemaphores(); !r) {
        return r.error();
    }
    if (auto r = renderer->createDepthBuffer(); !r) {
        return r.error();
    }

    log::info("Frame renderer ready ({} frames in flight, {} per-image semaphores)", kFramesInFlight,
              renderer->renderFinished_.size());
    return renderer;
}

Result<void> FrameRenderer::createSyncObjects() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = kFramesInFlight * 2; // scene + overlay per slot

    VkCommandBuffer buffers[kFramesInFlight * 2] = {};
    if (VkResult r = vkAllocateCommandBuffers(device_->handle(), &allocInfo, buffers); r != VK_SUCCESS) {
        return Error{std::format("vkAllocateCommandBuffers failed ({})", static_cast<int>(r))};
    }

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // first wait must pass

    for (std::uint32_t i = 0; i < kFramesInFlight; ++i) {
        frames_[i].commandBuffer = buffers[i];
        frames_[i].overlayCommandBuffer = buffers[kFramesInFlight + i];
        if (VkResult r =
                vkCreateSemaphore(device_->handle(), &semaphoreInfo, nullptr, &frames_[i].imageAvailable);
            r != VK_SUCCESS) {
            return Error{std::format("vkCreateSemaphore failed ({})", static_cast<int>(r))};
        }
        if (VkResult r = vkCreateFence(device_->handle(), &fenceInfo, nullptr, &frames_[i].inFlight);
            r != VK_SUCCESS) {
            return Error{std::format("vkCreateFence failed ({})", static_cast<int>(r))};
        }
    }
    return {};
}

Result<void> FrameRenderer::createImageSemaphores() {
    VkSemaphoreCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    renderFinished_.resize(swapchain_->images().size(), VK_NULL_HANDLE);
    for (auto& semaphore : renderFinished_) {
        if (VkResult r = vkCreateSemaphore(device_->handle(), &info, nullptr, &semaphore);
            r != VK_SUCCESS) {
            return Error{std::format("vkCreateSemaphore failed ({})", static_cast<int>(r))};
        }
    }
    return {};
}

Result<void> FrameRenderer::createDepthBuffer() {
    auto depthResult = Image::create(*device_, {
                                                   .width = swapchain_->width(),
                                                   .height = swapchain_->height(),
                                                   .format = kFormatD32Sfloat,
                                                   .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
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
                                                  .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                                           VK_IMAGE_USAGE_SAMPLED_BIT,
                                              });
        if (!result) {
            return Error{std::format("G-buffer target {}: {}", i, result.error().message)};
        }
        gbuffer_[i] = std::move(result).value();
        deferredTable_->writeSampledImage(28 + i, 0, gbuffer_[i]->view());
    }
    // The HDR scene-color target the composite pass renders into when the
    // batch carries a post pipeline; the post pass Loads it (binding 35).
    auto sceneColorResult = Image::create(*device_, {
                                                        .width = swapchain_->width(),
                                                        .height = swapchain_->height(),
                                                        .format = kSceneColorFormat,
                                                        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                                                 VK_IMAGE_USAGE_SAMPLED_BIT,
                                                    });
    if (!sceneColorResult) {
        return Error{std::format("Scene-color target: {}", sceneColorResult.error().message)};
    }
    sceneColor_ = std::move(sceneColorResult).value();
    deferredTable_->writeSampledImage(35, 0, sceneColor_->view());
    return {};
}

Result<void> FrameRenderer::setDeferredTargets(DescriptorTable* table) {
    deferredTable_ = table;
    if (table == nullptr) {
        gbuffer_ = {};
        sceneColor_.reset();
        return {};
    }
    return createGBuffer();
}

void FrameRenderer::destroyImageSemaphores() {
    for (VkSemaphore semaphore : renderFinished_) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_->handle(), semaphore, nullptr);
        }
    }
    renderFinished_.clear();
}

void FrameRenderer::resize(std::uint32_t width, std::uint32_t height) {
    pendingWidth_ = width;
    pendingHeight_ = height;
    resizeRequested_ = true;
}

void FrameRenderer::waitIdle() const {
    if (device_) {
        vkDeviceWaitIdle(device_->handle());
    }
}

Result<void> FrameRenderer::recreateSwapchain() {
    const std::uint32_t width = pendingWidth_ != 0 ? pendingWidth_ : swapchain_->width();
    const std::uint32_t height = pendingHeight_ != 0 ? pendingHeight_ : swapchain_->height();
    resizeRequested_ = false;

    const std::size_t previousImageCount = swapchain_->images().size();
    if (auto r = swapchain_->recreate(width, height); !r) {
        return r.error();
    }
    // Static recordings bake image handles and extent; rebuild lazily.
    invalidateStatic();
    // Semaphores are indexed by swapchain image; a changed count needs a
    // fresh set. The device is idle after recreate(), so this is safe.
    if (swapchain_->images().size() != previousImageCount) {
        destroyImageSemaphores();
        if (auto r = createImageSemaphores(); !r) {
            return r.error();
        }
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

Result<void> FrameRenderer::record(VkCommandBuffer cmd, std::uint32_t imageIndex,
                                   std::uint32_t slot, const DrawBatch* batch,
                                   bool reusable) const {
    REND_PROFILE_ZONE("RecordScene");
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = reusable ? 0 : VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (VkResult r = vkBeginCommandBuffer(cmd, &begin); r != VK_SUCCESS) {
        return Error{std::format("vkBeginCommandBuffer failed ({})", static_cast<int>(r))};
    }

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
        VkMemoryBarrier2 toSkin{};
        toSkin.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        // COMPUTE in the source: the previous frame's OBB refine dispatch
        // reads the pool too (WAR — execution ordering is enough).
        toSkin.srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
                              VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        toSkin.srcAccessMask = 0;
        toSkin.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        toSkin.dstAccessMask =
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo skinDependency{};
        skinDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        skinDependency.memoryBarrierCount = 1;
        skinDependency.pMemoryBarriers = &toSkin;
        vkCmdPipelineBarrier2(cmd, &skinDependency);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, batch->skinPipeline->handle());
        if (batch->descriptors != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    batch->skinPipeline->layout(), 0, 1, &batch->descriptors, 0,
                                    nullptr);
        }
        for (const DrawBatch::SkinDispatch& dispatch : batch->skinDispatches) {
            auto push = dispatch.push;
            push[DrawBatch::kSkinSlotPushIndex] = slot;
            vkCmdPushConstants(cmd, batch->skinPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(push), push.data());
            vkCmdDispatch(cmd, (dispatch.vertexCount + 63) / 64, 1, 1);
        }

        VkMemoryBarrier2 fromSkin{};
        fromSkin.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        fromSkin.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        fromSkin.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        // COMPUTE in the destination: this frame's OBB refine dispatch
        // reads the pool (disjoint regions from the posed writes, but the
        // hazard tracking is buffer-wide).
        fromSkin.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        fromSkin.dstAccessMask =
            VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        skinDependency.pMemoryBarriers = &fromSkin;
        vkCmdPipelineBarrier2(cmd, &skinDependency);
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
        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.memoryBarrierCount = 1;

        VkMemoryBarrier2 toBuild{};
        toBuild.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        toBuild.srcStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        toBuild.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        toBuild.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        toBuild.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
                                VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        dependency.pMemoryBarriers = &toBuild;
        vkCmdPipelineBarrier2(cmd, &dependency);

        if (blasRefit) {
            batch->refitBlas->recordRefit(cmd, batch->refitGeometries[slot]);

            // The TLAS rebuild reads the refitted BLAS AABBs.
            VkMemoryBarrier2 blasToTlas{};
            blasToTlas.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            blasToTlas.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            blasToTlas.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            blasToTlas.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            blasToTlas.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                       VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            dependency.pMemoryBarriers = &blasToTlas;
            vkCmdPipelineBarrier2(cmd, &dependency);
        }

        if (batch->tlasRebuild) {
            batch->tlasRebuild->recordRebuild(cmd, slot);
        }

        VkMemoryBarrier2 toTrace{};
        toTrace.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        toTrace.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        toTrace.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        toTrace.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        toTrace.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        dependency.pMemoryBarriers = &toTrace;
        vkCmdPipelineBarrier2(cmd, &dependency);
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
                               batch->occlusionVisibility != VK_NULL_HANDLE &&
                               (cullFlags & 4u) != 0;
        VkDependencyInfo cullDependency{};
        cullDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        if (occlusion) {
            // This slot's visibility region was read by the PREVIOUS
            // frame's cull dispatch — order that read before the clear.
            VkMemoryBarrier2 computeToClear{};
            computeToClear.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            computeToClear.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            computeToClear.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            computeToClear.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
            computeToClear.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            cullDependency.memoryBarrierCount = 1;
            cullDependency.pMemoryBarriers = &computeToClear;
            vkCmdPipelineBarrier2(cmd, &cullDependency);
            // Zero this slot's region for the proxy pass at the end of
            // THIS frame; the cull dispatch below reads the OTHER slot's.
            vkCmdFillBuffer(cmd, batch->occlusionVisibility,
                            slot * batch->occlusionRegionStride,
                            batch->occlusionRegionStride, 0);
        }
        vkCmdFillBuffer(cmd, batch->count, slot * batch->countRegionStride,
                        batch->countRegionStride, 0);

        // The fills must land before the dispatch reads/increments, and —
        // with occlusion — the previous frame's proxy-pass fragment
        // stores into the other slot's region must be visible too. With
        // OBB refinement the previous frame's refine writes (COMPUTE) and
        // proxy-VS OBB reads (VERTEX, WAR) join the source scope: this
        // frame's refine dispatch re-reads the counter and overwrites
        // rows the previous frame's consumers looked at.
        const bool obbRefine = batch->obbRefinePipeline != nullptr;
        VkMemoryBarrier2 fillToCompute{};
        fillToCompute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        fillToCompute.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT |
                                     (occlusion ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                                : VkPipelineStageFlags2{0}) |
                                     (obbRefine ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                                      VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
                                                : VkPipelineStageFlags2{0});
        fillToCompute.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT |
                                      (occlusion || obbRefine
                                           ? VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                                           : VkAccessFlags2{0});
        fillToCompute.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        fillToCompute.dstAccessMask =
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        cullDependency.memoryBarrierCount = 1;
        cullDependency.pMemoryBarriers = &fillToCompute;
        vkCmdPipelineBarrier2(cmd, &cullDependency);

        if (obbRefine) {
            // Self-terminating OBB refinement: claims the next
            // obbRefineGroups entries from binding 33's counter and fits
            // their oriented boxes; exits immediately once every entry
            // has been claimed. Baked like everything here — the counter
            // is the only state, so no recording ever changes.
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              batch->obbRefinePipeline->handle());
            if (batch->descriptors != VK_NULL_HANDLE) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        batch->obbRefinePipeline->layout(), 0, 1,
                                        &batch->descriptors, 0, nullptr);
            }
            const std::uint32_t obbPush[3] = {
                batch->drawCount, slot,
                static_cast<std::uint32_t>(batch->indirectRegionStride /
                                           sizeof(DrawIndexedIndirect))};
            vkCmdPushConstants(cmd, batch->obbRefinePipeline->layout(),
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(obbPush), obbPush);
            vkCmdDispatch(cmd, batch->obbRefineGroups, 1, 1);

            // The cull dispatch reads the rows the refine pass just wrote.
            VkMemoryBarrier2 refineToCull{};
            refineToCull.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            refineToCull.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            refineToCull.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            refineToCull.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            refineToCull.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            cullDependency.pMemoryBarriers = &refineToCull;
            vkCmdPipelineBarrier2(cmd, &cullDependency);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, batch->cullPipeline->handle());
        if (batch->descriptors != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    batch->cullPipeline->layout(), 0, 1, &batch->descriptors, 0,
                                    nullptr);
        }
        // capacity = the per-slot region stride in entries; drawCount can
        // grow at runtime (model loads) while the regions stay put. The
        // fifth word is the LOD screen-size factor, a float in disguise.
        std::uint32_t push[5] = {
            batch->drawCount, slot,
            static_cast<std::uint32_t>(batch->indirectRegionStride /
                                       sizeof(DrawIndexedIndirect)),
            cullFlags, 0};
        std::memcpy(&push[4], &batch->lodFactor, sizeof(float));
        vkCmdPushConstants(cmd, batch->cullPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), push);
        vkCmdDispatch(cmd, (batch->drawCount + 63) / 64, 1, 1);

        VkMemoryBarrier2 computeToDraw{};
        computeToDraw.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        computeToDraw.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        computeToDraw.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        // The vertex stage also reads what the cull pass wrote: partially
        // visible draws' surviving instance rows land in the rows buffer's
        // per-slot scratch regions.
        computeToDraw.dstStageMask =
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        computeToDraw.dstAccessMask =
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        cullDependency.pMemoryBarriers = &computeToDraw;
        vkCmdPipelineBarrier2(cmd, &cullDependency);
    }

    // Binds the batch's geometry/descriptors and emits its draw stream
    // with the given pipeline — shared by the shadow, main and blend
    // passes. The cascade index only matters to the shadow pipeline.
    // stream: 0 = the visibility-only list the shadow passes draw, 1 = the
    // frustum-culled opaque list (when the batch carries one), 2 = the
    // frustum-culled transparent list.
    auto bindAndDraw = [&](const Pipeline& p, std::uint32_t cascade, std::uint32_t stream) {
        const VkDeviceSize zero = 0;
        if (batch->descriptors != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout(), 0, 1,
                                    &batch->descriptors, 0, nullptr);
        }
        vkCmdBindVertexBuffers(cmd, 0, 1, &batch->geometry, &zero);
        vkCmdBindIndexBuffer(cmd, batch->geometry, 0, VK_INDEX_TYPE_UINT32);
        const std::uint32_t push[2] = {slot, cascade};
        vkCmdPushConstants(cmd, p.layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(push), push);
        // Counter layout per slot: [0] shadow, [1] opaque, [2] scratch
        // rows, [3] transparent.
        VkBuffer streamBuffer = batch->indirect;
        std::uint64_t countOffset = 0;
        if (stream == 1 && batch->sceneIndirect != VK_NULL_HANDLE) {
            streamBuffer = batch->sceneIndirect;
            countOffset = 1 * sizeof(std::uint32_t);
        } else if (stream == 2) {
            streamBuffer = batch->transparentIndirect;
            countOffset = 3 * sizeof(std::uint32_t);
        }
        switch (batch->mode) {
        case DrawSubmitMode::IndirectCount:
            vkCmdDrawIndexedIndirectCount(
                cmd, streamBuffer, slot * batch->indirectRegionStride, batch->count,
                slot * batch->countRegionStride + countOffset, batch->drawCount,
                sizeof(DrawIndexedIndirect));
            break;
        case DrawSubmitMode::Indirect:
            vkCmdDrawIndexedIndirect(cmd, batch->indirect, slot * batch->indirectRegionStride,
                                     batch->drawCount, sizeof(DrawIndexedIndirect));
            break;
        case DrawSubmitMode::Direct:
            for (std::uint32_t i = 0; i < batch->drawCount; ++i) {
                const DrawIndexedIndirect& draw = batch->cpuDraws[i];
                if (draw.instanceCount != 0) {
                    vkCmdDrawIndexed(cmd, draw.indexCount, draw.instanceCount, draw.firstIndex,
                                     draw.vertexOffset, draw.firstInstance);
                }
            }
            break;
        }
    };

    VkDependencyInfo shadowDependency{};
    shadowDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    shadowDependency.imageMemoryBarrierCount = 1;

    if (batch && (!rtDraw || fogCascades) && batch->shadowPipeline && batch->cascadeCount > 0) {
        for (std::uint32_t c = 0; c < batch->cascadeCount; ++c) {
            const Image* map = batch->shadowCascades[c];
            // Depth-only pass from the light's view. Contents are cleared,
            // so the old layout is UNDEFINED; the barrier orders against
            // the previous frame's sampling and depth writes.
            VkImageMemoryBarrier2 toShadowWrite = imageBarrier(
                map->handle(), VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
            toShadowWrite.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            shadowDependency.pImageMemoryBarriers = &toShadowWrite;
            vkCmdPipelineBarrier2(cmd, &shadowDependency);

            VkRenderingAttachmentInfo shadowDepth{};
            shadowDepth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            shadowDepth.imageView = map->view();
            shadowDepth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            shadowDepth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            shadowDepth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            shadowDepth.clearValue.depthStencil = {1.0f, 0};

            const VkExtent2D shadowExtent{map->width(), map->height()};
            VkRenderingInfo shadowRendering{};
            shadowRendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            shadowRendering.renderArea = {{0, 0}, shadowExtent};
            shadowRendering.layerCount = 1;
            shadowRendering.pDepthAttachment = &shadowDepth;
            vkCmdBeginRendering(cmd, &shadowRendering);

            const VkViewport shadowViewport{0.0f,
                                            0.0f,
                                            static_cast<float>(shadowExtent.width),
                                            static_cast<float>(shadowExtent.height),
                                            0.0f,
                                            1.0f};
            const VkRect2D shadowScissor{{0, 0}, shadowExtent};
            vkCmdSetViewport(cmd, 0, 1, &shadowViewport);
            vkCmdSetScissor(cmd, 0, 1, &shadowScissor);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              batch->shadowPipeline->handle());
            bindAndDraw(*batch->shadowPipeline, c, 0);
            vkCmdEndRendering(cmd);

            // Written depth becomes sampleable by the main pass's fragments.
            VkImageMemoryBarrier2 toShadowRead = imageBarrier(
                map->handle(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            toShadowRead.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            shadowDependency.pImageMemoryBarriers = &toShadowRead;
            vkCmdPipelineBarrier2(cmd, &shadowDependency);
        }
    }

    if (rasterScene) {
        // G-buffer pass: rasterize the opaque stream's surface attributes
        // into the four screen-sized targets plus depth. Target contents
        // are cleared, so old layouts are UNDEFINED; the barriers order
        // against the previous frame's lighting-pass reads.
        std::array<VkImageMemoryBarrier2, kGBufferTargets> toGBufferWrite{};
        for (std::uint32_t i = 0; i < kGBufferTargets; ++i) {
            toGBufferWrite[i] = imageBarrier(
                gbuffer_[i]->handle(), VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        }
        // Depth is cleared here and consumed by the composite pass's
        // depth-tested draws (sky/transparent/proxy); the barrier orders
        // against the previous frame's depth accesses.
        VkImageMemoryBarrier2 toDepthWrite = imageBarrier(
            depth_->handle(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        toDepthWrite.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

        std::array<VkImageMemoryBarrier2, kGBufferTargets + 1> gbufferBarriers{};
        std::copy(toGBufferWrite.begin(), toGBufferWrite.end(), gbufferBarriers.begin());
        gbufferBarriers.back() = toDepthWrite;
        VkDependencyInfo gbufferDependency{};
        gbufferDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        gbufferDependency.imageMemoryBarrierCount =
            static_cast<std::uint32_t>(gbufferBarriers.size());
        gbufferDependency.pImageMemoryBarriers = gbufferBarriers.data();
        vkCmdPipelineBarrier2(cmd, &gbufferDependency);

        std::array<VkRenderingAttachmentInfo, kGBufferTargets> gbufferColors{};
        for (std::uint32_t i = 0; i < kGBufferTargets; ++i) {
            gbufferColors[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            gbufferColors[i].imageView = gbuffer_[i]->view();
            gbufferColors[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            gbufferColors[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            gbufferColors[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            // All-zero clears: the zeroed view-depth target is the
            // lighting pass's background sentinel.
            gbufferColors[i].clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        }
        VkRenderingAttachmentInfo gbufferDepth{};
        gbufferDepth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        gbufferDepth.imageView = depth_->view();
        gbufferDepth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        gbufferDepth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        gbufferDepth.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // composite pass loads it
        gbufferDepth.clearValue.depthStencil = {1.0f, 0};

        const VkExtent2D gbufferExtent{swapchain_->width(), swapchain_->height()};
        VkRenderingInfo gbufferRendering{};
        gbufferRendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        gbufferRendering.renderArea = {{0, 0}, gbufferExtent};
        gbufferRendering.layerCount = 1;
        gbufferRendering.colorAttachmentCount = static_cast<std::uint32_t>(gbufferColors.size());
        gbufferRendering.pColorAttachments = gbufferColors.data();
        gbufferRendering.pDepthAttachment = &gbufferDepth;
        vkCmdBeginRendering(cmd, &gbufferRendering);

        const VkViewport gbufferViewport{0.0f,
                                         0.0f,
                                         static_cast<float>(gbufferExtent.width),
                                         static_cast<float>(gbufferExtent.height),
                                         0.0f,
                                         1.0f};
        const VkRect2D gbufferScissor{{0, 0}, gbufferExtent};
        vkCmdSetViewport(cmd, 0, 1, &gbufferViewport);
        vkCmdSetScissor(cmd, 0, 1, &gbufferScissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, batch->gbufferPipeline->handle());
        bindAndDraw(*batch->gbufferPipeline, 0, 1);
        vkCmdEndRendering(cmd);

        // Written targets become sampleable by the lighting triangle, and
        // the stored depth becomes testable by the composite pass's
        // depth-tested draws (sky/transparent/proxy).
        std::array<VkImageMemoryBarrier2, kGBufferTargets + 1> toRead{};
        for (std::uint32_t i = 0; i < kGBufferTargets; ++i) {
            toRead[i] = imageBarrier(gbuffer_[i]->handle(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
        VkImageMemoryBarrier2 depthToTest = imageBarrier(
            depth_->handle(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
        depthToTest.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        toRead.back() = depthToTest;
        VkDependencyInfo toReadDependency{};
        toReadDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        toReadDependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(toRead.size());
        toReadDependency.pImageMemoryBarriers = toRead.data();
        vkCmdPipelineBarrier2(cmd, &toReadDependency);
    }

    VkImage image = swapchain_->images()[imageIndex];

    // Post mode: the composite pass renders into the HDR scene-color
    // target and a final fullscreen pass (exposure + tonemap) maps it to
    // the swapchain. Every composite-pass pipeline must then declare
    // kSceneColorFormat. Without a post pipeline the composite pass
    // targets the swapchain directly, exactly as before.
    const bool post = batch && batch->postPipeline != nullptr && sceneColor_ != nullptr;

    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = 1;
    if (post) {
        // Scene color is cleared each frame (old layout UNDEFINED); order
        // against the previous frame's post-pass sampled reads.
        VkImageMemoryBarrier2 toSceneColor = imageBarrier(
            sceneColor_->handle(), VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        dependency.pImageMemoryBarriers = &toSceneColor;
        vkCmdPipelineBarrier2(cmd, &dependency);
    } else {
        VkImageMemoryBarrier2 toColor = imageBarrier(
            image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        dependency.pImageMemoryBarriers = &toColor;
        vkCmdPipelineBarrier2(cmd, &dependency);
    }

    VkRenderingAttachmentInfo color{};
    color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color.imageView = post ? sceneColor_->view() : swapchain_->imageViews()[imageIndex];
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{clearColor_[0], clearColor_[1], clearColor_[2], clearColor_[3]}};

    VkRenderingAttachmentInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth.imageView = depth_ ? depth_->view() : VK_NULL_HANDLE;
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    // Loads the G-buffer pass's stored depth — the sky/transparent/proxy
    // draws test against it; nothing here writes it.
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    const VkExtent2D extent{swapchain_->width(), swapchain_->height()};
    VkRenderingInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea = {{0, 0}, extent};
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &color;
    // Depth only when rasterizing a batch: the attachment set must match
    // the pipeline's declared depthFormat (traced primary needs none).
    rendering.pDepthAttachment = rasterScene ? &depth : nullptr;
    vkCmdBeginRendering(cmd, &rendering);

    const VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent.width),
                              static_cast<float>(extent.height), 0.0f, 1.0f};
    const VkRect2D scissor{{0, 0}, extent};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    if (rtDraw) {
        // Fullscreen traced pass: every pixel fires a camera ray in the
        // fragment shader; camera/light/slot data flow exactly as in the
        // raster path, so static recordings survive camera motion here too.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          batch->rtPrimaryPipeline->handle());
        if (batch->descriptors != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    batch->rtPrimaryPipeline->layout(), 0, 1,
                                    &batch->descriptors, 0, nullptr);
        }
        const std::uint32_t push[2] = {slot, 0};
        vkCmdPushConstants(cmd, batch->rtPrimaryPipeline->layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(push), push);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    } else if (batch) {
        // Lighting: one fullscreen triangle Loads the G-buffer (bindings
        // 28-31) and shades every covered pixel; the opaque stream was
        // already rasterized in the G-buffer pass. Background pixels
        // discard, keeping the baked clear color for the sky pass to
        // overdraw.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          batch->lightingPipeline->handle());
        if (batch->descriptors != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    batch->lightingPipeline->layout(), 0, 1,
                                    &batch->descriptors, 0, nullptr);
        }
        const std::uint32_t lightingPush[2] = {slot, 0};
        vkCmdPushConstants(cmd, batch->lightingPipeline->layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(lightingPush), lightingPush);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        {
            // Sky pass: fullscreen triangle at the far plane, depth test
            // only — paints the per-slot skyColor over background pixels
            // before the transparents blend on top of it.
            if (batch->skyPipeline != nullptr) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  batch->skyPipeline->handle());
                if (batch->descriptors != VK_NULL_HANDLE) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            batch->skyPipeline->layout(), 0, 1,
                                            &batch->descriptors, 0, nullptr);
                }
                const std::uint32_t skyPush[2] = {slot, 0};
                vkCmdPushConstants(cmd, batch->skyPipeline->layout(),
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(skyPush), skyPush);
                vkCmdDraw(cmd, 3, 1, 0, 0);
            }
            // Transparency pass: same rendering pass, blend pipeline,
            // depth write off — the cull shader routed these entries out
            // of the opaque stream. Unsorted for now (single-layer glass
            // is fine; stacked transparents may blend out of order).
            if (batch->transparentPipeline && batch->transparentIndirect != VK_NULL_HANDLE &&
                batch->mode == DrawSubmitMode::IndirectCount) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  batch->transparentPipeline->handle());
                bindAndDraw(*batch->transparentPipeline, 0, 2);
            }
            // Occlusion proxy pass: every template's world AABB as an
            // instanced cube against the frame's finished depth (test
            // only, color masked); survivors mark the visibility buffer
            // the NEXT frame's cull dispatch consumes. Same rendering
            // pass, so the scene's depth writes are already ordered.
            if (batch->occlusionPipeline != nullptr && (batch->cullFlags & 4u) != 0 &&
                batch->mode == DrawSubmitMode::IndirectCount) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  batch->occlusionPipeline->handle());
                if (batch->descriptors != VK_NULL_HANDLE) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            batch->occlusionPipeline->layout(), 0, 1,
                                            &batch->descriptors, 0, nullptr);
                }
                const std::uint32_t proxyPush[2] = {
                    slot, static_cast<std::uint32_t>(batch->indirectRegionStride /
                                                     sizeof(DrawIndexedIndirect))};
                vkCmdPushConstants(cmd, batch->occlusionPipeline->layout(),
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(proxyPush), proxyPush);
                vkCmdDraw(cmd, 36, batch->drawCount, 0, 0);
                // Per-instance proxy pass: one box per canonical instance
                // row (dead/scene rows emit degenerate geometry). Same
                // push constants and PS; marks the region's per-instance
                // visibility slots.
                if (batch->occlusionInstancePipeline != nullptr &&
                    batch->occlusionInstanceRows > 0) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                      batch->occlusionInstancePipeline->handle());
                    if (batch->descriptors != VK_NULL_HANDLE) {
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                batch->occlusionInstancePipeline->layout(), 0,
                                                1, &batch->descriptors, 0, nullptr);
                    }
                    vkCmdPushConstants(cmd, batch->occlusionInstancePipeline->layout(),
                                       VK_SHADER_STAGE_VERTEX_BIT |
                                           VK_SHADER_STAGE_FRAGMENT_BIT,
                                       0, sizeof(proxyPush), proxyPush);
                    vkCmdDraw(cmd, 36, batch->occlusionInstanceRows, 0, 0);
                }
            }
            // Occlusion-box debug overlay: the same instanced AABB cubes
            // as translucent color (identical depth state, so the tinted
            // fragments are exactly the proxy pass's survivors). Not
            // gated on cullFlags bit 2 — the boxes are inspectable with
            // occlusion culling toggled off.
            if (batch->occlusionDebugPipeline != nullptr &&
                batch->mode == DrawSubmitMode::IndirectCount) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  batch->occlusionDebugPipeline->handle());
                if (batch->descriptors != VK_NULL_HANDLE) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            batch->occlusionDebugPipeline->layout(), 0, 1,
                                            &batch->descriptors, 0, nullptr);
                }
                const std::uint32_t debugPush[2] = {
                    slot, static_cast<std::uint32_t>(batch->indirectRegionStride /
                                                     sizeof(DrawIndexedIndirect))};
                vkCmdPushConstants(cmd, batch->occlusionDebugPipeline->layout(),
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(debugPush), debugPush);
                vkCmdDraw(cmd, 36, batch->drawCount, 0, 0);
                // The instanced models' boxes, same overlay styling.
                if (batch->occlusionInstanceDebugPipeline != nullptr &&
                    batch->occlusionInstanceRows > 0) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                      batch->occlusionInstanceDebugPipeline->handle());
                    if (batch->descriptors != VK_NULL_HANDLE) {
                        vkCmdBindDescriptorSets(
                            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            batch->occlusionInstanceDebugPipeline->layout(), 0, 1,
                            &batch->descriptors, 0, nullptr);
                    }
                    vkCmdPushConstants(cmd, batch->occlusionInstanceDebugPipeline->layout(),
                                       VK_SHADER_STAGE_VERTEX_BIT |
                                           VK_SHADER_STAGE_FRAGMENT_BIT,
                                       0, sizeof(debugPush), debugPush);
                    vkCmdDraw(cmd, 36, batch->occlusionInstanceRows, 0, 0);
                }
            }
        }
    }
    // No batch: nothing draws — the cleared swapchain image (plus the
    // overlay pass) is the whole frame.

    vkCmdEndRendering(cmd);

    if (post) {
        // Scene color becomes sampleable, the swapchain image becomes the
        // real render target, and one fullscreen triangle maps HDR scene
        // color to it (exposure + tonemap from the light buffer).
        std::array<VkImageMemoryBarrier2, 2> toPost{};
        toPost[0] = imageBarrier(sceneColor_->handle(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        toPost[1] = imageBarrier(image, VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        VkDependencyInfo postDependency{};
        postDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        postDependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(toPost.size());
        postDependency.pImageMemoryBarriers = toPost.data();
        vkCmdPipelineBarrier2(cmd, &postDependency);

        VkRenderingAttachmentInfo postColor{};
        postColor.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        postColor.imageView = swapchain_->imageViews()[imageIndex];
        postColor.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // The triangle covers every pixel unconditionally.
        postColor.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        postColor.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo postRendering{};
        postRendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        postRendering.renderArea = {{0, 0}, extent};
        postRendering.layerCount = 1;
        postRendering.colorAttachmentCount = 1;
        postRendering.pColorAttachments = &postColor;
        vkCmdBeginRendering(cmd, &postRendering);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, batch->postPipeline->handle());
        if (batch->descriptors != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    batch->postPipeline->layout(), 0, 1, &batch->descriptors, 0,
                                    nullptr);
        }
        const std::uint32_t postPush[2] = {slot, 0};
        vkCmdPushConstants(cmd, batch->postPipeline->layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(postPush), postPush);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
    }

    // The image stays in COLOR_ATTACHMENT_OPTIMAL: the per-frame overlay
    // command buffer draws the UI on top and owns the present transition.
    if (VkResult r = vkEndCommandBuffer(cmd); r != VK_SUCCESS) {
        return Error{std::format("vkEndCommandBuffer failed ({})", static_cast<int>(r))};
    }
    return {};
}

Result<void> FrameRenderer::recordOverlay(VkCommandBuffer cmd, std::uint32_t imageIndex) const {
    REND_PROFILE_ZONE("RecordOverlay");
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (VkResult r = vkBeginCommandBuffer(cmd, &begin); r != VK_SUCCESS) {
        return Error{std::format("vkBeginCommandBuffer (overlay) failed ({})", static_cast<int>(r))};
    }

    VkImage image = swapchain_->images()[imageIndex];
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = 1;

    if (overlayRecorder_) {
        // Order against the scene buffer's color writes (same submission,
        // no layout change) before loading the attachment.
        VkImageMemoryBarrier2 sceneToOverlay = imageBarrier(
            image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        dependency.pImageMemoryBarriers = &sceneToOverlay;
        vkCmdPipelineBarrier2(cmd, &dependency);

        VkRenderingAttachmentInfo color{};
        color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageView = swapchain_->imageViews()[imageIndex];
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        const VkExtent2D extent{swapchain_->width(), swapchain_->height()};
        VkRenderingInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering.renderArea = {{0, 0}, extent};
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &color;
        vkCmdBeginRendering(cmd, &rendering);

        const VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent.width),
                                  static_cast<float>(extent.height), 0.0f, 1.0f};
        const VkRect2D scissor{{0, 0}, extent};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        overlayRecorder_(cmd);

        vkCmdEndRendering(cmd);
    }

    VkImageMemoryBarrier2 toPresent = imageBarrier(
        image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0);
    dependency.pImageMemoryBarriers = &toPresent;
    vkCmdPipelineBarrier2(cmd, &dependency);

    if (VkResult r = vkEndCommandBuffer(cmd); r != VK_SUCCESS) {
        return Error{std::format("vkEndCommandBuffer (overlay) failed ({})", static_cast<int>(r))};
    }
    return {};
}

void FrameRenderer::setStaticRecording(bool enabled) {
    if (staticEnabled_ == enabled) {
        return;
    }
    staticEnabled_ = enabled;
    invalidateStatic();
}

void FrameRenderer::invalidateStatic() {
    if (!staticBuffers_.empty()) {
        // In-flight frames may still be executing these buffers; freeing a
        // pending command buffer is invalid and wedges the driver.
        vkDeviceWaitIdle(device_->handle());
        vkFreeCommandBuffers(device_->handle(), commandPool_,
                             static_cast<std::uint32_t>(staticBuffers_.size()),
                             staticBuffers_.data());
        staticBuffers_.clear();
    }
    staticValid_ = false;
}

Result<void> FrameRenderer::prerecordStatic(const DrawBatch* batch) {
    REND_PROFILE_ZONE("PrerecordStatic");
    invalidateStatic();

    const auto imageCount = static_cast<std::uint32_t>(swapchain_->images().size());
    staticBuffers_.resize(std::size_t{kFramesInFlight} * imageCount, VK_NULL_HANDLE);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<std::uint32_t>(staticBuffers_.size());
    if (VkResult r = vkAllocateCommandBuffers(device_->handle(), &allocInfo, staticBuffers_.data());
        r != VK_SUCCESS) {
        staticBuffers_.clear();
        return Error{std::format("vkAllocateCommandBuffers (static) failed ({})",
                                 static_cast<int>(r))};
    }

    for (std::uint32_t slot = 0; slot < kFramesInFlight; ++slot) {
        for (std::uint32_t image = 0; image < imageCount; ++image) {
            if (auto r = record(staticBuffers_[std::size_t{slot} * imageCount + image], image, slot,
                                batch, /*reusable=*/true);
                !r) {
                invalidateStatic();
                return r.error();
            }
        }
    }
    staticValid_ = true;
    ++stats_.prerecords;
    log::info("Static command buffers recorded: {} ({} slots x {} images)", staticBuffers_.size(),
              kFramesInFlight, imageCount);
    return {};
}

Result<void> FrameRenderer::waitFrameSlot() {
    return waitForFence(frames_[frameIndex_].inFlight, "Previous frame");
}

FrameRenderer::Stats FrameRenderer::takeStats() {
    Stats out = stats_;
    stats_ = {};
    return out;
}

Result<void> FrameRenderer::waitForFence(VkFence fence, const char* what) const {
    REND_PROFILE_ZONE("WaitFence");
    for (int attempt = 0;; ++attempt) {
        const VkResult waited = vkWaitForFences(device_->handle(), 1, &fence, VK_TRUE, kWaitTimeoutNs);
        if (waited == VK_SUCCESS) {
            return {};
        }
        if (waited != VK_TIMEOUT) {
            return Error{std::format("vkWaitForFences failed ({})", static_cast<int>(waited))};
        }
        if (attempt + 1 >= kMaxStalledWaits) {
            return Error{std::format("{} fence never signalled; the GPU appears stalled", what)};
        }
        log::warn("{} still pending after {} s", what, (attempt + 1) * 2);
    }
}

Result<void> FrameRenderer::drawFrame(const DrawBatch* batch) {
    REND_PROFILE_ZONE("DrawFrame");
    if (resizeRequested_) {
        if (pendingWidth_ == 0 || pendingHeight_ == 0) {
            return {}; // minimized: nothing to present to
        }
        if (auto r = recreateSwapchain(); !r) {
            return r.error();
        }
    }

    FrameData& frame = frames_[frameIndex_];
    if (auto r = waitForFence(frame.inFlight, "Previous frame"); !r) {
        return r.error();
    }

    std::uint32_t imageIndex = 0;
    // On VK_TIMEOUT no semaphore is signalled, so the same one is reusable.
    VkResult acquired = VK_SUCCESS;
    {
        REND_PROFILE_ZONE("AcquireImage");
        acquired = vkAcquireNextImageKHR(device_->handle(), swapchain_->handle(), kWaitTimeoutNs,
                                         frame.imageAvailable, VK_NULL_HANDLE, &imageIndex);
    }
    if (acquired == VK_TIMEOUT || acquired == VK_NOT_READY) {
        log::warn("No swapchain image available within {} s", kWaitTimeoutNs / 1'000'000'000ull);
        return {};
    }
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
        resizeRequested_ = true;
        return {}; // the semaphore was not signalled; retry next frame
    }
    if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
        return Error{std::format("vkAcquireNextImageKHR failed ({})", static_cast<int>(acquired))};
    }

    vkResetFences(device_->handle(), 1, &frame.inFlight);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (staticEnabled_) {
        if (!staticValid_) {
            if (auto r = prerecordStatic(batch); !r) {
                return r.error();
            }
        }
        cmd = staticBuffers_[std::size_t{frameIndex_} * swapchain_->images().size() + imageIndex];
    } else {
        const auto recordStart = std::chrono::steady_clock::now();
        cmd = frame.commandBuffer;
        vkResetCommandBuffer(cmd, 0);
        if (auto r = record(cmd, imageIndex, frameIndex_, batch, /*reusable=*/false);
            !r) {
            return r.error();
        }
        stats_.recordMicros += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                                  recordStart)
                .count());
    }
    ++stats_.frames;

    // The overlay tail (UI + present transition) is re-recorded every
    // frame; the scene buffer above may be a static recording.
    vkResetCommandBuffer(frame.overlayCommandBuffer, 0);
    if (auto r = recordOverlay(frame.overlayCommandBuffer, imageIndex); !r) {
        return r.error();
    }

    VkSemaphoreSubmitInfo waitInfo{};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    waitInfo.semaphore = frame.imageAvailable;
    waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSemaphoreSubmitInfo signalInfo{};
    signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalInfo.semaphore = renderFinished_[imageIndex];
    signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    std::array<VkCommandBufferSubmitInfo, 2> commandInfos{};
    commandInfos[0].sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandInfos[0].commandBuffer = cmd;
    commandInfos[1].sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandInfos[1].commandBuffer = frame.overlayCommandBuffer;

    VkSubmitInfo2 submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &waitInfo;
    submit.commandBufferInfoCount = static_cast<std::uint32_t>(commandInfos.size());
    submit.pCommandBufferInfos = commandInfos.data();
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signalInfo;

    {
        REND_PROFILE_ZONE("QueueSubmit");
        if (VkResult r = vkQueueSubmit2(device_->graphicsQueue().queue, 1, &submit, frame.inFlight);
            r != VK_SUCCESS) {
            return Error{std::format("vkQueueSubmit2 failed ({})", static_cast<int>(r))};
        }
    }

    VkSwapchainKHR swapchainHandle = swapchain_->handle();
    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &renderFinished_[imageIndex];
    present.swapchainCount = 1;
    present.pSwapchains = &swapchainHandle;
    present.pImageIndices = &imageIndex;

    VkResult presented = VK_SUCCESS;
    {
        REND_PROFILE_ZONE("Present");
        presented = vkQueuePresentKHR(device_->graphicsQueue().queue, &present);
    }
    if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR ||
        acquired == VK_SUBOPTIMAL_KHR) {
        resizeRequested_ = true;
    } else if (presented != VK_SUCCESS) {
        return Error{std::format("vkQueuePresentKHR failed ({})", static_cast<int>(presented))};
    }

    frameIndex_ = (frameIndex_ + 1) % kFramesInFlight;
    return {};
}

FrameRenderer::~FrameRenderer() {
    if (!device_ || device_->handle() == VK_NULL_HANDLE) {
        return;
    }
    vkDeviceWaitIdle(device_->handle());

    destroyImageSemaphores();
    for (FrameData& frame : frames_) {
        if (frame.imageAvailable != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_->handle(), frame.imageAvailable, nullptr);
        }
        if (frame.inFlight != VK_NULL_HANDLE) {
            vkDestroyFence(device_->handle(), frame.inFlight, nullptr);
        }
    }
    if (commandPool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_->handle(), commandPool_, nullptr);
    }
    log::info("Frame renderer destroyed");
}

} // namespace rend::gpu

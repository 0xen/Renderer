#include "rend/gpu/frame_renderer.h"

#include "rend/core/log.h"
#include "rend/core/profile.h"
#include "rend/gpu/device.h"
#include "rend/gpu/image.h"
#include "rend/gpu/pipeline.h"
#include "rend/gpu/swapchain.h"

#include <volk.h>

#include <chrono>

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
    return {};
}

Result<void> FrameRenderer::record(VkCommandBuffer cmd, std::uint32_t imageIndex,
                                   std::uint32_t slot, const Pipeline& pipeline,
                                   const DrawBatch* batch, bool reusable) const {
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

    if (batch && batch->skinPipeline && !batch->skinDispatches.empty()) {
        // Pose animated vertices into this slot's pool regions. Order
        // against the previous frame's vertex fetches, then make the
        // writes visible to every consumer of the pool this frame.
        VkMemoryBarrier2 toSkin{};
        toSkin.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        toSkin.srcStageMask =
            VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
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
        fromSkin.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        fromSkin.dstAccessMask =
            VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        skinDependency.pMemoryBarriers = &fromSkin;
        vkCmdPipelineBarrier2(cmd, &skinDependency);
    }

    if (batch && batch->refitBlas && batch->refitTlas && slot < batch->refitGeometries.size() &&
        !batch->refitGeometries[slot].empty()) {
        // Refit the BLAS from this slot's posed vertices, then the TLAS so
        // its instance AABBs follow. The entry barrier makes the skin
        // writes visible to the build AND — queue-scoped, so it reaches
        // across submissions — orders it after the previous frame in
        // flight's ray queries reading the same structures.
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

        batch->refitBlas->recordRefit(cmd, batch->refitGeometries[slot]);

        VkMemoryBarrier2 blasToTlas{};
        blasToTlas.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        blasToTlas.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        blasToTlas.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        blasToTlas.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        blasToTlas.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                   VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        dependency.pMemoryBarriers = &blasToTlas;
        vkCmdPipelineBarrier2(cmd, &dependency);

        batch->refitTlas->recordRefit(cmd);

        VkMemoryBarrier2 toTrace{};
        toTrace.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        toTrace.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        toTrace.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        toTrace.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        toTrace.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        dependency.pMemoryBarriers = &toTrace;
        vkCmdPipelineBarrier2(cmd, &dependency);
    }

    if (batch && !rtDraw && batch->cullPipeline && batch->mode == DrawSubmitMode::IndirectCount) {
        // GPU compaction: zero the slot's draw count, run one thread per
        // template, then make the writes visible to the indirect fetch.
        vkCmdFillBuffer(cmd, batch->count, slot * batch->countRegionStride,
                        sizeof(std::uint32_t), 0);

        VkMemoryBarrier2 fillToCompute{};
        fillToCompute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        fillToCompute.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        fillToCompute.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        fillToCompute.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        fillToCompute.dstAccessMask =
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo cullDependency{};
        cullDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        cullDependency.memoryBarrierCount = 1;
        cullDependency.pMemoryBarriers = &fillToCompute;
        vkCmdPipelineBarrier2(cmd, &cullDependency);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, batch->cullPipeline->handle());
        if (batch->descriptors != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    batch->cullPipeline->layout(), 0, 1, &batch->descriptors, 0,
                                    nullptr);
        }
        const std::uint32_t push[2] = {batch->drawCount, slot};
        vkCmdPushConstants(cmd, batch->cullPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), push);
        vkCmdDispatch(cmd, (batch->drawCount + 63) / 64, 1, 1);

        VkMemoryBarrier2 computeToDraw{};
        computeToDraw.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        computeToDraw.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        computeToDraw.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        computeToDraw.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        computeToDraw.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        cullDependency.pMemoryBarriers = &computeToDraw;
        vkCmdPipelineBarrier2(cmd, &cullDependency);
    }

    // Binds the batch's geometry/descriptors and emits its draw stream
    // with the given pipeline — shared by the shadow and main passes. The
    // cascade index only matters to the shadow pipeline.
    auto bindAndDraw = [&](const Pipeline& p, std::uint32_t cascade) {
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
        switch (batch->mode) {
        case DrawSubmitMode::IndirectCount:
            vkCmdDrawIndexedIndirectCount(cmd, batch->indirect,
                                          slot * batch->indirectRegionStride, batch->count,
                                          slot * batch->countRegionStride, batch->drawCount,
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

    if (batch && !rtDraw && batch->shadowPipeline && batch->cascadeCount > 0) {
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
            bindAndDraw(*batch->shadowPipeline, c);
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

    VkImage image = swapchain_->images()[imageIndex];

    VkImageMemoryBarrier2 toColor =
        imageBarrier(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &toColor;
    vkCmdPipelineBarrier2(cmd, &dependency);

    if (batch && !rtDraw) {
        // Depth contents are cleared each frame, so the previous frame's
        // layout is irrelevant (UNDEFINED); the barrier orders against the
        // prior frame's depth accesses.
        VkImageMemoryBarrier2 toDepth = imageBarrier(
            depth_->handle(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        toDepth.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        dependency.pImageMemoryBarriers = &toDepth;
        vkCmdPipelineBarrier2(cmd, &dependency);
    }

    VkRenderingAttachmentInfo color{};
    color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color.imageView = swapchain_->imageViews()[imageIndex];
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{clearColor_[0], clearColor_[1], clearColor_[2], clearColor_[3]}};

    VkRenderingAttachmentInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth.imageView = depth_ ? depth_->view() : VK_NULL_HANDLE;
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.clearValue.depthStencil = {1.0f, 0};

    const VkExtent2D extent{swapchain_->width(), swapchain_->height()};
    VkRenderingInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea = {{0, 0}, extent};
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &color;
    // Depth only when rasterizing a batch: the attachment set must match
    // the pipeline's declared depthFormat (traced primary needs none).
    rendering.pDepthAttachment = (batch && !rtDraw) ? &depth : nullptr;
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
    } else {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle());
        if (batch) {
            // The whole scene: geometry pool bound once, one indirect stream.
            bindAndDraw(pipeline, 0);
        } else {
            vkCmdDraw(cmd, 3, 1, 0, 0);
        }
    }

    vkCmdEndRendering(cmd);

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

Result<void> FrameRenderer::prerecordStatic(const Pipeline& pipeline, const DrawBatch* batch) {
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
                                pipeline, batch, /*reusable=*/true);
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

Result<void> FrameRenderer::drawFrame(const Pipeline& pipeline, const DrawBatch* batch) {
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
            if (auto r = prerecordStatic(pipeline, batch); !r) {
                return r.error();
            }
        }
        cmd = staticBuffers_[std::size_t{frameIndex_} * swapchain_->images().size() + imageIndex];
    } else {
        const auto recordStart = std::chrono::steady_clock::now();
        cmd = frame.commandBuffer;
        vkResetCommandBuffer(cmd, 0);
        if (auto r = record(cmd, imageIndex, frameIndex_, pipeline, batch, /*reusable=*/false);
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

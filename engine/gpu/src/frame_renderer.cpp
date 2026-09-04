#include "rend/gpu/frame_renderer.h"

#include "rend/core/log.h"
#include "rend/gpu/device.h"
#include "rend/gpu/pipeline.h"
#include "rend/gpu/swapchain.h"

#include <volk.h>

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

    log::info("Frame renderer ready ({} frames in flight, {} per-image semaphores)", kFramesInFlight,
              renderer->renderFinished_.size());
    return renderer;
}

Result<void> FrameRenderer::createSyncObjects() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = kFramesInFlight;

    VkCommandBuffer buffers[kFramesInFlight] = {};
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
    // Semaphores are indexed by swapchain image; a changed count needs a
    // fresh set. The device is idle after recreate(), so this is safe.
    if (swapchain_->images().size() != previousImageCount) {
        destroyImageSemaphores();
        if (auto r = createImageSemaphores(); !r) {
            return r.error();
        }
    }
    return {};
}

Result<void> FrameRenderer::record(VkCommandBuffer cmd, std::uint32_t imageIndex,
                                   const Pipeline& pipeline) const {
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (VkResult r = vkBeginCommandBuffer(cmd, &begin); r != VK_SUCCESS) {
        return Error{std::format("vkBeginCommandBuffer failed ({})", static_cast<int>(r))};
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

    VkRenderingAttachmentInfo color{};
    color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color.imageView = swapchain_->imageViews()[imageIndex];
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{clearColor_[0], clearColor_[1], clearColor_[2], clearColor_[3]}};

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

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle());
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRendering(cmd);

    VkImageMemoryBarrier2 toPresent = imageBarrier(
        image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0);
    dependency.pImageMemoryBarriers = &toPresent;
    vkCmdPipelineBarrier2(cmd, &dependency);

    if (VkResult r = vkEndCommandBuffer(cmd); r != VK_SUCCESS) {
        return Error{std::format("vkEndCommandBuffer failed ({})", static_cast<int>(r))};
    }
    return {};
}

Result<void> FrameRenderer::waitForFence(VkFence fence, const char* what) const {
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

Result<void> FrameRenderer::drawFrame(const Pipeline& pipeline) {
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
    VkResult acquired = vkAcquireNextImageKHR(device_->handle(), swapchain_->handle(), kWaitTimeoutNs,
                                              frame.imageAvailable, VK_NULL_HANDLE, &imageIndex);
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
    vkResetCommandBuffer(frame.commandBuffer, 0);
    if (auto r = record(frame.commandBuffer, imageIndex, pipeline); !r) {
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

    VkCommandBufferSubmitInfo commandInfo{};
    commandInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandInfo.commandBuffer = frame.commandBuffer;

    VkSubmitInfo2 submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &waitInfo;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &commandInfo;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signalInfo;

    if (VkResult r = vkQueueSubmit2(device_->graphicsQueue().queue, 1, &submit, frame.inFlight);
        r != VK_SUCCESS) {
        return Error{std::format("vkQueueSubmit2 failed ({})", static_cast<int>(r))};
    }

    VkSwapchainKHR swapchainHandle = swapchain_->handle();
    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &renderFinished_[imageIndex];
    present.swapchainCount = 1;
    present.pSwapchains = &swapchainHandle;
    present.pImageIndices = &imageIndex;

    const VkResult presented = vkQueuePresentKHR(device_->graphicsQueue().queue, &present);
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

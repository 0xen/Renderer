#include "vulkan_types.h"

#include "rend/core/log.h"
#include "rend/core/profile.h"

#include <chrono>
#include <format>

namespace rend::gpu {

namespace {

// The frame loop never blocks forever: a fence or acquire that stalls this
// long is reported and retried, so a wedged driver stays diagnosable (and
// the process stays killable) instead of silently hanging.
constexpr std::uint64_t kWaitTimeoutNs = 2'000'000'000ull;
constexpr int kMaxStalledWaits = 5;

} // namespace

Result<std::unique_ptr<FrameRenderer>> VulkanFrameRenderer::create(const Device& deviceBase,
                                                                   Swapchain& swapchainBase) {
    const VulkanDevice& device = vk(deviceBase);
    auto renderer = std::unique_ptr<VulkanFrameRenderer>(new VulkanFrameRenderer());
    renderer->device_ = &device;
    renderer->swapchain_ = &swapchainBase;
    renderer->vkDevice_ = &device;
    renderer->vkSwapchain_ = &vk(swapchainBase);

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
    return std::unique_ptr<FrameRenderer>(std::move(renderer));
}

Result<void> VulkanFrameRenderer::createSyncObjects() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = kFramesInFlight * 2; // scene + overlay per slot

    VkCommandBuffer buffers[kFramesInFlight * 2] = {};
    if (VkResult r = vkAllocateCommandBuffers(vkDevice_->handle(), &allocInfo, buffers);
        r != VK_SUCCESS) {
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
        if (VkResult r = vkCreateSemaphore(vkDevice_->handle(), &semaphoreInfo, nullptr,
                                           &frames_[i].imageAvailable);
            r != VK_SUCCESS) {
            return Error{std::format("vkCreateSemaphore failed ({})", static_cast<int>(r))};
        }
        if (VkResult r = vkCreateFence(vkDevice_->handle(), &fenceInfo, nullptr, &frames_[i].inFlight);
            r != VK_SUCCESS) {
            return Error{std::format("vkCreateFence failed ({})", static_cast<int>(r))};
        }
    }
    return {};
}

Result<void> VulkanFrameRenderer::createImageSemaphores() {
    VkSemaphoreCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    renderFinished_.resize(vkSwapchain_->images().size(), VK_NULL_HANDLE);
    for (auto& semaphore : renderFinished_) {
        if (VkResult r = vkCreateSemaphore(vkDevice_->handle(), &info, nullptr, &semaphore);
            r != VK_SUCCESS) {
            return Error{std::format("vkCreateSemaphore failed ({})", static_cast<int>(r))};
        }
    }
    return {};
}

void VulkanFrameRenderer::destroyImageSemaphores() {
    for (VkSemaphore semaphore : renderFinished_) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(vkDevice_->handle(), semaphore, nullptr);
        }
    }
    renderFinished_.clear();
}

void VulkanFrameRenderer::waitIdle() const {
    if (vkDevice_) {
        vkDeviceWaitIdle(vkDevice_->handle());
    }
}

Result<void> VulkanFrameRenderer::onSwapchainRecreated(std::uint32_t previousImageCount) {
    // Semaphores are indexed by swapchain image; a changed count needs a
    // fresh set. The device is idle after recreate(), so this is safe.
    if (vkSwapchain_->images().size() != previousImageCount) {
        destroyImageSemaphores();
        if (auto r = createImageSemaphores(); !r) {
            return r.error();
        }
    }
    return {};
}

Result<void> VulkanFrameRenderer::record(VkCommandBuffer cmd, std::uint32_t imageIndex,
                                         std::uint32_t slot, const DrawBatch* batch,
                                         bool reusable) const {
    REND_PROFILE_ZONE("RecordScene");
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = reusable ? 0 : VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (VkResult r = vkBeginCommandBuffer(cmd, &begin); r != VK_SUCCESS) {
        return Error{std::format("vkBeginCommandBuffer failed ({})", static_cast<int>(r))};
    }

    VulkanCommandContext ctx(cmd);
    recordFrame(ctx, imageIndex, slot, batch);

    // The image stays in COLOR_ATTACHMENT_OPTIMAL: the per-frame overlay
    // command buffer draws the UI on top and owns the present transition.
    if (VkResult r = vkEndCommandBuffer(cmd); r != VK_SUCCESS) {
        return Error{std::format("vkEndCommandBuffer failed ({})", static_cast<int>(r))};
    }
    return {};
}

Result<void> VulkanFrameRenderer::recordOverlay(VkCommandBuffer cmd, std::uint32_t imageIndex) const {
    REND_PROFILE_ZONE("RecordOverlay");
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (VkResult r = vkBeginCommandBuffer(cmd, &begin); r != VK_SUCCESS) {
        return Error{std::format("vkBeginCommandBuffer (overlay) failed ({})", static_cast<int>(r))};
    }

    VulkanCommandContext ctx(cmd);
    recordOverlayFrame(ctx, imageIndex);

    if (VkResult r = vkEndCommandBuffer(cmd); r != VK_SUCCESS) {
        return Error{std::format("vkEndCommandBuffer (overlay) failed ({})", static_cast<int>(r))};
    }
    return {};
}

void VulkanFrameRenderer::invalidateStatic() {
    if (!staticBuffers_.empty()) {
        // In-flight frames may still be executing these buffers; freeing a
        // pending command buffer is invalid and wedges the driver.
        vkDeviceWaitIdle(vkDevice_->handle());
        vkFreeCommandBuffers(vkDevice_->handle(), commandPool_,
                             static_cast<std::uint32_t>(staticBuffers_.size()),
                             staticBuffers_.data());
        staticBuffers_.clear();
    }
    staticValid_ = false;
}

Result<void> VulkanFrameRenderer::prerecordStatic(const DrawBatch* batch) {
    REND_PROFILE_ZONE("PrerecordStatic");
    invalidateStatic();

    const auto imageCount = static_cast<std::uint32_t>(vkSwapchain_->images().size());
    staticBuffers_.resize(std::size_t{kFramesInFlight} * imageCount, VK_NULL_HANDLE);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<std::uint32_t>(staticBuffers_.size());
    if (VkResult r = vkAllocateCommandBuffers(vkDevice_->handle(), &allocInfo, staticBuffers_.data());
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

Result<void> VulkanFrameRenderer::waitFrameSlot() {
    return waitForFence(frames_[frameIndex_].inFlight, "Previous frame");
}

Result<void> VulkanFrameRenderer::waitForFence(VkFence fence, const char* what) const {
    REND_PROFILE_ZONE("WaitFence");
    for (int attempt = 0;; ++attempt) {
        const VkResult waited =
            vkWaitForFences(vkDevice_->handle(), 1, &fence, VK_TRUE, kWaitTimeoutNs);
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

Result<void> VulkanFrameRenderer::drawFrame(const DrawBatch* batch) {
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
        acquired = vkAcquireNextImageKHR(vkDevice_->handle(), vkSwapchain_->handle(), kWaitTimeoutNs,
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

    vkResetFences(vkDevice_->handle(), 1, &frame.inFlight);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (staticEnabled_) {
        if (!staticValid_) {
            if (auto r = prerecordStatic(batch); !r) {
                return r.error();
            }
        }
        cmd = staticBuffers_[std::size_t{frameIndex_} * vkSwapchain_->images().size() + imageIndex];
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
        if (VkResult r = vkQueueSubmit2(vkDevice_->graphicsQueue().queue, 1, &submit, frame.inFlight);
            r != VK_SUCCESS) {
            return Error{std::format("vkQueueSubmit2 failed ({})", static_cast<int>(r))};
        }
    }

    VkSwapchainKHR swapchainHandle = vkSwapchain_->handle();
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
        presented = vkQueuePresentKHR(vkDevice_->graphicsQueue().queue, &present);
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

VulkanFrameRenderer::~VulkanFrameRenderer() {
    if (!vkDevice_ || vkDevice_->handle() == VK_NULL_HANDLE) {
        return;
    }
    vkDeviceWaitIdle(vkDevice_->handle());

    destroyImageSemaphores();
    for (FrameData& frame : frames_) {
        if (frame.imageAvailable != VK_NULL_HANDLE) {
            vkDestroySemaphore(vkDevice_->handle(), frame.imageAvailable, nullptr);
        }
        if (frame.inFlight != VK_NULL_HANDLE) {
            vkDestroyFence(vkDevice_->handle(), frame.inFlight, nullptr);
        }
    }
    if (commandPool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(vkDevice_->handle(), commandPool_, nullptr);
    }
    log::info("Frame renderer destroyed");
}

} // namespace rend::gpu

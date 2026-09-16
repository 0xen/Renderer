#include "rend/gpu/transfer.h"

#include "vulkan/vulkan_types.h"
#include "d3d12/d3d12_types.h"

#include "rend/core/log.h"
#include "rend/gpu/buffer.h"
#include "rend/gpu/device.h"

#include <volk.h>

#include <cstring>
#include <format>

namespace rend::gpu {

namespace {
constexpr std::uint64_t kInitialStagingBytes = 4ull * 1024 * 1024;
constexpr std::uint64_t kFenceTimeoutNs = 10ull * 1000 * 1000 * 1000;
} // namespace

Result<std::unique_ptr<TransferContext>> VulkanTransferContext::create(const Device& deviceBase) {
    const VulkanDevice& device = vk(deviceBase);
    auto context = std::unique_ptr<VulkanTransferContext>(new VulkanTransferContext());
    context->device_ = &device;

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = device.transferQueue().familyIndex;
    if (VkResult r = vkCreateCommandPool(device.handle(), &poolInfo, nullptr, &context->pool_);
        r != VK_SUCCESS) {
        return Error{std::format("vkCreateCommandPool failed ({})", static_cast<int>(r))};
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = context->pool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    if (VkResult r = vkAllocateCommandBuffers(device.handle(), &allocInfo, &context->cmd_);
        r != VK_SUCCESS) {
        return Error{std::format("vkAllocateCommandBuffers failed ({})", static_cast<int>(r))};
    }

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (VkResult r = vkCreateFence(device.handle(), &fenceInfo, nullptr, &context->fence_);
        r != VK_SUCCESS) {
        return Error{std::format("vkCreateFence failed ({})", static_cast<int>(r))};
    }
    return std::unique_ptr<TransferContext>(std::move(context));
}

VulkanTransferContext::~VulkanTransferContext() {
    if (!device_) {
        return;
    }
    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_->handle(), fence_, nullptr);
    }
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_->handle(), pool_, nullptr);
    }
}

Result<void> VulkanTransferContext::ensureStagingCapacity(std::uint64_t required) {
    if (staging_ && staging_->size() >= required) {
        return {};
    }
    std::uint64_t capacity = staging_ ? staging_->size() : kInitialStagingBytes;
    while (capacity < required) {
        capacity *= 2;
    }
    auto grown = Buffer::create(*device_, {
                                              .size = capacity,
                                              .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                              .location = MemoryLocation::HostVisible,
                                          });
    if (!grown) {
        return grown.error();
    }
    // Pending copies reference staging offsets only, so the bytes must move
    // with the reallocation.
    if (staging_ && stagingUsed_ > 0) {
        std::memcpy(grown.value()->mapped(), staging_->mapped(), stagingUsed_);
    }
    staging_ = std::move(grown).value();
    return {};
}

Result<void> VulkanTransferContext::stage(const Buffer& dst, std::uint64_t dstOffset, const void* data,
                                    std::uint64_t size) {
    if (size == 0) {
        return {};
    }
    if (dstOffset + size > dst.size()) {
        return Error{std::format("Staged copy overruns destination ({} + {} > {})", dstOffset,
                                 size, dst.size())};
    }
    if (auto r = ensureStagingCapacity(stagingUsed_ + size); !r) {
        return r;
    }
    std::memcpy(static_cast<std::byte*>(staging_->mapped()) + stagingUsed_, data, size);
    pending_.push_back({.dst = &dst, .srcOffset = stagingUsed_, .dstOffset = dstOffset, .size = size});
    stagingUsed_ += size;
    return {};
}

Result<void> VulkanTransferContext::flush() {
    if (pending_.empty()) {
        return {};
    }

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (VkResult r = vkBeginCommandBuffer(cmd_, &begin); r != VK_SUCCESS) {
        return Error{std::format("vkBeginCommandBuffer failed ({})", static_cast<int>(r))};
    }
    for (const PendingCopy& copy : pending_) {
        VkBufferCopy region{.srcOffset = copy.srcOffset,
                            .dstOffset = copy.dstOffset,
                            .size = copy.size};
        vkCmdCopyBuffer(cmd_, vk(*staging_).handle(), vk(*copy.dst).handle(), 1, &region);
    }
    if (VkResult r = vkEndCommandBuffer(cmd_); r != VK_SUCCESS) {
        return Error{std::format("vkEndCommandBuffer failed ({})", static_cast<int>(r))};
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd_;
    if (VkResult r = vkQueueSubmit(device_->transferQueue().queue, 1, &submit, fence_);
        r != VK_SUCCESS) {
        return Error{std::format("vkQueueSubmit failed ({})", static_cast<int>(r))};
    }
    const VkResult waited = vkWaitForFences(device_->handle(), 1, &fence_, VK_TRUE, kFenceTimeoutNs);
    if (waited != VK_SUCCESS) {
        return Error{std::format("Transfer fence wait failed ({})", static_cast<int>(waited))};
    }
    vkResetFences(device_->handle(), 1, &fence_);
    vkResetCommandBuffer(cmd_, 0);

    log::trace("Transfer flushed: {} copies, {} bytes", pending_.size(), stagingUsed_);
    pending_.clear();
    stagingUsed_ = 0;
    return {};
}

Result<std::unique_ptr<TransferContext>> TransferContext::create(const Device& device) {
    switch (device.api()) {
    case Api::Vulkan: return VulkanTransferContext::create(device);
    case Api::D3D12: return D3D12TransferContext::create(device);
    }
    return Error{std::format("{} backend: TransferContext not implemented", apiName(device.api()))};
}

} // namespace rend::gpu

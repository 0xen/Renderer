#include "rend/gpu/texture_uploader.h"

#include "rend/core/profile.h"
#include "rend/gpu/buffer.h"
#include "rend/gpu/device.h"

#include <volk.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <format>

namespace rend::gpu {

namespace {

constexpr std::uint64_t kFenceTimeoutNs = 30ull * 1000 * 1000 * 1000;
constexpr std::uint32_t kFormatR8G8B8A8Srgb = 43;  // VK_FORMAT_R8G8B8A8_SRGB
constexpr std::uint32_t kFormatR8G8B8A8Unorm = 37; // VK_FORMAT_R8G8B8A8_UNORM

VkImageMemoryBarrier2 mipBarrier(VkImage image, std::uint32_t baseMip, std::uint32_t mipCount,
                                 VkImageLayout oldLayout, VkImageLayout newLayout,
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
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount, 0, 1};
    return barrier;
}

void applyBarrier(VkCommandBuffer cmd, const VkImageMemoryBarrier2& barrier) {
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dependency);
}

} // namespace

Result<std::unique_ptr<TextureUploader>> TextureUploader::create(const Device& device) {
    auto uploader = std::unique_ptr<TextureUploader>(new TextureUploader());
    uploader->device_ = &device;

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = device.graphicsQueue().familyIndex;
    if (VkResult r = vkCreateCommandPool(device.handle(), &poolInfo, nullptr, &uploader->pool_);
        r != VK_SUCCESS) {
        return Error{std::format("vkCreateCommandPool failed ({})", static_cast<int>(r))};
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = uploader->pool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    if (VkResult r = vkAllocateCommandBuffers(device.handle(), &allocInfo, &uploader->cmd_);
        r != VK_SUCCESS) {
        return Error{std::format("vkAllocateCommandBuffers failed ({})", static_cast<int>(r))};
    }

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (VkResult r = vkCreateFence(device.handle(), &fenceInfo, nullptr, &uploader->fence_);
        r != VK_SUCCESS) {
        return Error{std::format("vkCreateFence failed ({})", static_cast<int>(r))};
    }
    return uploader;
}

TextureUploader::~TextureUploader() {
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

Result<void> TextureUploader::ensureStagingCapacity(std::uint64_t required) {
    if (staging_ && staging_->size() >= required) {
        return {};
    }
    auto grown = Buffer::create(*device_, {
                                              .size = std::bit_ceil(required),
                                              .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                              .location = MemoryLocation::HostVisible,
                                          });
    if (!grown) {
        return grown.error();
    }
    staging_ = std::move(grown).value();
    return {};
}

Result<std::unique_ptr<Image>> TextureUploader::upload(std::uint32_t width, std::uint32_t height,
                                                       const void* rgba8, bool srgb) {
    REND_PROFILE_ZONE("TextureUpload");
    const std::uint64_t byteSize = 4ull * width * height;
    if (byteSize == 0 || !rgba8) {
        return Error{"Texture upload needs non-empty pixels"};
    }

    const std::uint32_t mipLevels =
        std::bit_width(std::max(width, height)); // floor(log2) + 1

    auto imageResult =
        Image::create(*device_, {
                                    .width = width,
                                    .height = height,
                                    .format = srgb ? kFormatR8G8B8A8Srgb : kFormatR8G8B8A8Unorm,
                                    .usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                             VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                    .mipLevels = mipLevels,
                                });
    if (!imageResult) {
        return imageResult.error();
    }
    auto image = std::move(imageResult).value();

    if (auto r = ensureStagingCapacity(byteSize); !r) {
        return r.error();
    }
    std::memcpy(staging_->mapped(), rgba8, byteSize);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (VkResult r = vkBeginCommandBuffer(cmd_, &begin); r != VK_SUCCESS) {
        return Error{std::format("vkBeginCommandBuffer failed ({})", static_cast<int>(r))};
    }

    applyBarrier(cmd_, mipBarrier(image->handle(), 0, mipLevels, VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                  VK_ACCESS_2_TRANSFER_WRITE_BIT));

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd_, staging_->handle(), image->handle(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Blit chain: each mip becomes the source for the next.
    std::int32_t mipWidth = static_cast<std::int32_t>(width);
    std::int32_t mipHeight = static_cast<std::int32_t>(height);
    for (std::uint32_t mip = 1; mip < mipLevels; ++mip) {
        applyBarrier(cmd_, mipBarrier(image->handle(), mip - 1, 1,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT));

        const std::int32_t nextWidth = std::max(mipWidth / 2, 1);
        const std::int32_t nextHeight = std::max(mipHeight / 2, 1);
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip - 1, 0, 1};
        blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
        blit.dstOffsets[1] = {nextWidth, nextHeight, 1};
        vkCmdBlitImage(cmd_, image->handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       image->handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                       VK_FILTER_LINEAR);
        mipWidth = nextWidth;
        mipHeight = nextHeight;
    }

    // Mips 0..n-2 are TRANSFER_SRC, the last is TRANSFER_DST; both end as
    // SHADER_READ_ONLY.
    if (mipLevels > 1) {
        applyBarrier(cmd_, mipBarrier(image->handle(), 0, mipLevels - 1,
                                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));
    }
    applyBarrier(cmd_, mipBarrier(image->handle(), mipLevels - 1, 1,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));

    if (VkResult r = vkEndCommandBuffer(cmd_); r != VK_SUCCESS) {
        return Error{std::format("vkEndCommandBuffer failed ({})", static_cast<int>(r))};
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd_;
    if (VkResult r = vkQueueSubmit(device_->graphicsQueue().queue, 1, &submit, fence_);
        r != VK_SUCCESS) {
        return Error{std::format("vkQueueSubmit failed ({})", static_cast<int>(r))};
    }
    if (VkResult r = vkWaitForFences(device_->handle(), 1, &fence_, VK_TRUE, kFenceTimeoutNs);
        r != VK_SUCCESS) {
        return Error{std::format("Texture upload fence wait failed ({})", static_cast<int>(r))};
    }
    vkResetFences(device_->handle(), 1, &fence_);
    vkResetCommandBuffer(cmd_, 0);
    return image;
}

} // namespace rend::gpu

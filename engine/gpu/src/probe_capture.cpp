#include "rend/gpu/probe_capture.h"

#include "vulkan/vulkan_types.h"

#include "rend/gpu/buffer.h"
#include "rend/gpu/descriptor_table.h"

#include "rend/core/profile.h"
#include "rend/gpu/device.h"
#include "rend/gpu/image.h"
#include "rend/gpu/pipeline.h"

#include <volk.h>

#include <algorithm>
#include <bit>
#include <format>

namespace rend::gpu {

namespace {

constexpr std::uint64_t kFenceTimeoutNs = 30ull * 1000 * 1000 * 1000;

VkImageMemoryBarrier2 cubeBarrier(VkImage image, std::uint32_t baseMip, std::uint32_t mipCount,
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
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount, 0, 6};
    return barrier;
}

void applyBarrier(VkCommandBuffer cmd, VkImageMemoryBarrier2 barrier) {
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dependency);
}

} // namespace

Result<std::unique_ptr<Image>> ProbeCapture::render(const Device& deviceBase,
                                                    const ProbeCaptureDesc& desc) {
    if (deviceBase.api() != Api::Vulkan) {
        // Records straight into a VkCommandBuffer (optional seam step 5
        // would move it onto CommandContext); callers treat the failure
        // as "no probe" and keep going.
        return Error{std::format("{} backend: ProbeCapture not implemented",
                                 apiName(deviceBase.api()))};
    }
    const VulkanDevice& device = vk(deviceBase);
    REND_PROFILE_ZONE("ProbeCapture");
    if (!desc.pipeline || !desc.draws || desc.drawCount == 0 || desc.faceSize == 0) {
        return Error{"Probe capture needs a pipeline and a non-empty draw list"};
    }

    const std::uint32_t fullChain = std::bit_width(desc.faceSize); // floor(log2) + 1
    const std::uint32_t mipLevels =
        desc.mipLevels == 0 ? fullChain : std::min(desc.mipLevels, fullChain);
    auto cubeResult = Image::create(device, {
                                                .width = desc.faceSize,
                                                .height = desc.faceSize,
                                                .format = desc.format,
                                                .usage = kImageUsageColorAttachment |
                                                         kImageUsageSampled |
                                                         kImageUsageTransferSrc |
                                                         kImageUsageTransferDst,
                                                .mipLevels = mipLevels,
                                                .cube = true,
                                            });
    if (!cubeResult) {
        return Error{std::format("Probe cubemap: {}", cubeResult.error().message)};
    }
    auto cube = std::move(cubeResult).value();

    auto depthResult = Image::create(device, {
                                                 .width = desc.faceSize,
                                                 .height = desc.faceSize,
                                                 .format = kFormatD32Sfloat,
                                                 .usage = kImageUsageDepthAttachment,
                                                 .depth = true,
                                             });
    if (!depthResult) {
        return Error{std::format("Probe depth buffer: {}", depthResult.error().message)};
    }
    auto depth = std::move(depthResult).value();

    // One-shot pool/buffer/fence, synchronous like TextureUploader: the
    // capture happens once at load, before the frame loop exists.
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = device.graphicsQueue().familyIndex;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (VkResult r = vkCreateCommandPool(device.handle(), &poolInfo, nullptr, &pool);
        r != VK_SUCCESS) {
        return Error{std::format("vkCreateCommandPool failed ({})", static_cast<int>(r))};
    }
    // Everything below funnels through this cleanup on both paths.
    auto cleanup = [&](VkFence fence) {
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device.handle(), fence, nullptr);
        }
        vkDestroyCommandPool(device.handle(), pool, nullptr);
    };

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (VkResult r = vkAllocateCommandBuffers(device.handle(), &allocInfo, &cmd);
        r != VK_SUCCESS) {
        cleanup(VK_NULL_HANDLE);
        return Error{std::format("vkAllocateCommandBuffers failed ({})", static_cast<int>(r))};
    }

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (VkResult r = vkBeginCommandBuffer(cmd, &begin); r != VK_SUCCESS) {
        cleanup(VK_NULL_HANDLE);
        return Error{std::format("vkBeginCommandBuffer failed ({})", static_cast<int>(r))};
    }

    applyBarrier(cmd, cubeBarrier(vk(*cube).handle(), 0, 1, VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_NONE, 0,
                                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT));

    const VkExtent2D extent{desc.faceSize, desc.faceSize};
    for (std::uint32_t face = 0; face < 6; ++face) {
        // Depth is cleared per face; UNDEFINED discards the previous face's
        // contents while the barrier orders the depth-write reuse.
        VkImageMemoryBarrier2 toDepth = cubeBarrier(
            vk(*depth).handle(), 0, 1, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        toDepth.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        applyBarrier(cmd, toDepth);

        VkRenderingAttachmentInfo color{};
        color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageView = vk(*cube).faceView(face);
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.clearValue.color = {{desc.clearColor[0], desc.clearColor[1], desc.clearColor[2],
                                   desc.clearColor[3]}};

        VkRenderingAttachmentInfo depthAttachment{};
        depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAttachment.imageView = vk(*depth).view();
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering.renderArea = {{0, 0}, extent};
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &color;
        rendering.pDepthAttachment = &depthAttachment;
        vkCmdBeginRendering(cmd, &rendering);

        const VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent.width),
                                  static_cast<float>(extent.height), 0.0f, 1.0f};
        const VkRect2D scissor{{0, 0}, extent};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk(*desc.pipeline).handle());
        if (desc.descriptors != nullptr) {
            const VkDescriptorSet set = vk(*desc.descriptors).set();
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    vk(*desc.pipeline).layout(), 0, 1, &set, 0, nullptr);
        }
        const VkDeviceSize zero = 0;
        const VkBuffer geometry = vk(*desc.geometry).handle();
        vkCmdBindVertexBuffers(cmd, 0, 1, &geometry, &zero);
        vkCmdBindIndexBuffer(cmd, geometry, 0, VK_INDEX_TYPE_UINT32);
        const std::uint32_t push[2] = {desc.cameraSlotBase + face, 0};
        vkCmdPushConstants(cmd, vk(*desc.pipeline).layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(push), push);
        // One-shot capture: plain CPU-side draws, no compaction machinery.
        for (std::uint32_t i = 0; i < desc.drawCount; ++i) {
            const DrawIndexedIndirect& draw = desc.draws[i];
            if (draw.instanceCount != 0) {
                vkCmdDrawIndexed(cmd, draw.indexCount, draw.instanceCount, draw.firstIndex,
                                 draw.vertexOffset, draw.firstInstance);
            }
        }
        vkCmdEndRendering(cmd);
    }

    // Blit chain over all six faces at once: mip 0 becomes the transfer
    // source, the rest fill top-down (TextureUploader's pattern, layered).
    applyBarrier(cmd, cubeBarrier(vk(*cube).handle(), 0, 1,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                  VK_ACCESS_2_TRANSFER_READ_BIT));
    if (mipLevels > 1) {
        applyBarrier(cmd, cubeBarrier(vk(*cube).handle(), 1, mipLevels - 1,
                                      VK_IMAGE_LAYOUT_UNDEFINED,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_NONE, 0,
                                      VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                      VK_ACCESS_2_TRANSFER_WRITE_BIT));
    }
    std::int32_t mipSize = static_cast<std::int32_t>(desc.faceSize);
    for (std::uint32_t mip = 1; mip < mipLevels; ++mip) {
        const std::int32_t nextSize = std::max(mipSize / 2, 1);
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip - 1, 0, 6};
        blit.srcOffsets[1] = {mipSize, mipSize, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 6};
        blit.dstOffsets[1] = {nextSize, nextSize, 1};
        vkCmdBlitImage(cmd, vk(*cube).handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk(*cube).handle(),
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        applyBarrier(cmd, cubeBarrier(vk(*cube).handle(), mip, 1,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                      VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                      VK_ACCESS_2_TRANSFER_READ_BIT));
        mipSize = nextSize;
    }
    applyBarrier(cmd, cubeBarrier(vk(*cube).handle(), 0, mipLevels,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));

    if (VkResult r = vkEndCommandBuffer(cmd); r != VK_SUCCESS) {
        cleanup(VK_NULL_HANDLE);
        return Error{std::format("vkEndCommandBuffer failed ({})", static_cast<int>(r))};
    }

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    if (VkResult r = vkCreateFence(device.handle(), &fenceInfo, nullptr, &fence);
        r != VK_SUCCESS) {
        cleanup(VK_NULL_HANDLE);
        return Error{std::format("vkCreateFence failed ({})", static_cast<int>(r))};
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (VkResult r = vkQueueSubmit(device.graphicsQueue().queue, 1, &submit, fence);
        r != VK_SUCCESS) {
        cleanup(fence);
        return Error{std::format("vkQueueSubmit failed ({})", static_cast<int>(r))};
    }
    if (VkResult r = vkWaitForFences(device.handle(), 1, &fence, VK_TRUE, kFenceTimeoutNs);
        r != VK_SUCCESS) {
        cleanup(fence);
        return Error{std::format("Probe capture fence wait failed ({})", static_cast<int>(r))};
    }
    cleanup(fence);
    return cube;
}

} // namespace rend::gpu

#include "rend/gpu/command_context.h"

#include "rend/gpu/buffer.h"
#include "rend/gpu/descriptor_table.h"
#include "rend/gpu/image.h"
#include "rend/gpu/pipeline.h"

#include <volk.h>

namespace rend::gpu {

namespace {

VkCommandBuffer vk(void* cmd) { return static_cast<VkCommandBuffer>(cmd); }

struct StateInfo {
    VkImageLayout layout;
    VkPipelineStageFlags2 stage;
    VkAccessFlags2 srcAccess; // what this state wrote
    VkAccessFlags2 dstAccess; // what this state will do
};

StateInfo stateInfo(ImageState state, bool depthAspect) {
    switch (state) {
    case ImageState::ColorAttachment:
        return {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT};
    case ImageState::DepthAttachment:
        return {VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
    case ImageState::ShaderRead:
        return {depthAspect ? VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL
                            : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                0, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
    case ImageState::Undefined:
    default:
        // Contents are discarded: only execution ordering against whatever
        // last touched the image (typically last frame's readers) matters.
        return {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, 0};
    }
}

VkPipelineStageFlags2 stageMask(Stage stages) {
    const auto bits = static_cast<std::uint32_t>(stages);
    VkPipelineStageFlags2 mask = 0;
    if (bits & static_cast<std::uint32_t>(Stage::ComputeShader)) {
        mask |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    }
    if (bits & static_cast<std::uint32_t>(Stage::VertexShader)) {
        mask |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    }
    if (bits & static_cast<std::uint32_t>(Stage::FragmentShader)) {
        mask |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    }
    if (bits & static_cast<std::uint32_t>(Stage::DrawIndirect)) {
        mask |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    }
    if (bits & static_cast<std::uint32_t>(Stage::VertexInput)) {
        mask |= VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
    }
    if (bits & static_cast<std::uint32_t>(Stage::ColorAttachment)) {
        mask |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    return mask;
}

VkPipelineStageFlags2 stageMask(PipelineStage stages) {
    const auto bits = static_cast<std::uint32_t>(stages);
    VkPipelineStageFlags2 mask = 0;
    auto map = [&](PipelineStage s, VkPipelineStageFlags2 vkBit) {
        if (bits & static_cast<std::uint32_t>(s)) {
            mask |= vkBit;
        }
    };
    map(PipelineStage::DrawIndirect, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
    map(PipelineStage::VertexAttributeInput, VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT);
    map(PipelineStage::IndexInput, VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT);
    map(PipelineStage::VertexShader, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
    map(PipelineStage::FragmentShader, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
    map(PipelineStage::EarlyFragmentTests, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT);
    map(PipelineStage::LateFragmentTests, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT);
    map(PipelineStage::ColorAttachmentOutput, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
    map(PipelineStage::ComputeShader, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    map(PipelineStage::Clear, VK_PIPELINE_STAGE_2_CLEAR_BIT);
    map(PipelineStage::AccelerationStructureBuild,
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
    map(PipelineStage::AllCommands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    return mask;
}

VkAccessFlags2 accessMask(Access access) {
    const auto bits = static_cast<std::uint32_t>(access);
    VkAccessFlags2 mask = 0;
    auto map = [&](Access a, VkAccessFlags2 vkBit) {
        if (bits & static_cast<std::uint32_t>(a)) {
            mask |= vkBit;
        }
    };
    map(Access::IndirectCommandRead, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    map(Access::VertexAttributeRead, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
    map(Access::ShaderRead, VK_ACCESS_2_SHADER_READ_BIT);
    map(Access::ShaderSampledRead, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    map(Access::ShaderStorageRead, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    map(Access::ShaderStorageWrite, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    map(Access::ColorAttachmentRead, VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);
    map(Access::ColorAttachmentWrite, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    map(Access::DepthStencilRead, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
    map(Access::DepthStencilWrite, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    map(Access::TransferWrite, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    map(Access::AccelerationStructureRead, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    map(Access::AccelerationStructureWrite, VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
    return mask;
}

VkImageLayout layoutOf(ImageLayout layout) {
    switch (layout) {
    case ImageLayout::ColorAttachment: return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case ImageLayout::DepthAttachment: return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    case ImageLayout::ShaderReadOnly: return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case ImageLayout::Present: return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    case ImageLayout::Undefined:
    default: return VK_IMAGE_LAYOUT_UNDEFINED;
    }
}

VkAttachmentLoadOp loadOp(LoadOp op) {
    switch (op) {
    case LoadOp::Load: return VK_ATTACHMENT_LOAD_OP_LOAD;
    case LoadOp::DontCare: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    case LoadOp::Clear:
    default: return VK_ATTACHMENT_LOAD_OP_CLEAR;
    }
}

} // namespace

void CommandContext::imageBarrier(const Image& image, ImageState from, ImageState to) {
    const bool depthAspect = isDepthFormat(image.format());
    const StateInfo src = stateInfo(from, depthAspect);
    const StateInfo dst = stateInfo(to, depthAspect);
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = src.stage;
    barrier.srcAccessMask = src.srcAccess;
    barrier.dstStageMask = dst.stage;
    barrier.dstAccessMask = dst.dstAccess;
    barrier.oldLayout = src.layout;
    barrier.newLayout = dst.layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.handle();
    barrier.subresourceRange = {
        static_cast<VkImageAspectFlags>(depthAspect ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                    : VK_IMAGE_ASPECT_COLOR_BIT),
        0, 1, 0, 1};
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(vk(cmd_), &dependency);
}

void CommandContext::memoryBarrier(Stage from, Stage to) {
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = stageMask(from);
    barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.dstStageMask = stageMask(to);
    barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(vk(cmd_), &dependency);
}

void CommandContext::barrier(std::span<const MemoryBarrierDesc> memory,
                             std::span<const ImageBarrierDesc> images) {
    std::vector<VkMemoryBarrier2> memoryBarriers(memory.size());
    for (std::size_t i = 0; i < memory.size(); ++i) {
        VkMemoryBarrier2& b = memoryBarriers[i];
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        b.srcStageMask = stageMask(memory[i].srcStage);
        b.srcAccessMask = accessMask(memory[i].srcAccess);
        b.dstStageMask = stageMask(memory[i].dstStage);
        b.dstAccessMask = accessMask(memory[i].dstAccess);
    }
    std::vector<VkImageMemoryBarrier2> imageBarriers(images.size());
    for (std::size_t i = 0; i < images.size(); ++i) {
        const ImageBarrierDesc& d = images[i];
        VkImageMemoryBarrier2& b = imageBarriers[i];
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = stageMask(d.srcStage);
        b.srcAccessMask = accessMask(d.srcAccess);
        b.dstStageMask = stageMask(d.dstStage);
        b.dstAccessMask = accessMask(d.dstAccess);
        b.oldLayout = layoutOf(d.oldLayout);
        b.newLayout = layoutOf(d.newLayout);
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = d.image->handle();
        b.subresourceRange = {
            static_cast<VkImageAspectFlags>(isDepthFormat(d.image->format())
                                                ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                : VK_IMAGE_ASPECT_COLOR_BIT),
            0, 1, 0, 1};
    }
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.memoryBarrierCount = static_cast<std::uint32_t>(memoryBarriers.size());
    dependency.pMemoryBarriers = memoryBarriers.data();
    dependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(imageBarriers.size());
    dependency.pImageMemoryBarriers = imageBarriers.data();
    vkCmdPipelineBarrier2(vk(cmd_), &dependency);
}

void CommandContext::fillBuffer(const Buffer& buffer, std::uint64_t offset, std::uint64_t size,
                                std::uint32_t value) {
    vkCmdFillBuffer(vk(cmd_), buffer.handle(), offset, size, value);
}

void CommandContext::drawIndexedIndirectCount(const Buffer& buffer, std::uint64_t offset,
                                              const Buffer& count, std::uint64_t countOffset,
                                              std::uint32_t maxDrawCount, std::uint32_t stride) {
    vkCmdDrawIndexedIndirectCount(vk(cmd_), buffer.handle(), offset, count.handle(), countOffset,
                                  maxDrawCount,
                                  stride == 0 ? sizeof(VkDrawIndexedIndirectCommand) : stride);
}

void CommandContext::beginRendering(const RenderingDesc& desc) {
    std::vector<VkRenderingAttachmentInfo> colors(desc.colors.size());
    for (std::size_t i = 0; i < colors.size(); ++i) {
        const ColorTarget& target = desc.colors[i];
        colors[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colors[i].imageView = target.image->view();
        colors[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colors[i].loadOp = loadOp(target.load);
        colors[i].storeOp = target.store ? VK_ATTACHMENT_STORE_OP_STORE
                                         : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colors[i].clearValue.color = {
            {target.clear[0], target.clear[1], target.clear[2], target.clear[3]}};
    }
    VkRenderingAttachmentInfo depth{};
    if (desc.depth) {
        depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth.imageView = desc.depth->image->view();
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = loadOp(desc.depth->load);
        depth.storeOp = desc.depth->store ? VK_ATTACHMENT_STORE_OP_STORE
                                          : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.clearValue.depthStencil = {desc.depth->clear, 0};
    }

    VkRenderingInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea = {{0, 0}, {desc.width, desc.height}};
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = static_cast<std::uint32_t>(colors.size());
    rendering.pColorAttachments = colors.data();
    rendering.pDepthAttachment = desc.depth ? &depth : nullptr;
    vkCmdBeginRendering(vk(cmd_), &rendering);

    setViewport(0.0f, 0.0f, static_cast<float>(desc.width), static_cast<float>(desc.height));
    setScissor(0, 0, desc.width, desc.height);
}

void CommandContext::endRendering() { vkCmdEndRendering(vk(cmd_)); }

void CommandContext::setViewport(float x, float y, float width, float height) {
    const VkViewport viewport{x, y, width, height, 0.0f, 1.0f};
    vkCmdSetViewport(vk(cmd_), 0, 1, &viewport);
}

void CommandContext::setScissor(std::int32_t x, std::int32_t y, std::uint32_t width,
                                std::uint32_t height) {
    const VkRect2D scissor{{x, y}, {width, height}};
    vkCmdSetScissor(vk(cmd_), 0, 1, &scissor);
}

void CommandContext::bindPipeline(const Pipeline& pipeline) {
    vkCmdBindPipeline(vk(cmd_),
                      pipeline.isCompute() ? VK_PIPELINE_BIND_POINT_COMPUTE
                                           : VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipeline.handle());
}

void CommandContext::bindDescriptorTable(const Pipeline& pipeline, const DescriptorTable& table) {
    const VkDescriptorSet set = table.set();
    vkCmdBindDescriptorSets(vk(cmd_),
                            pipeline.isCompute() ? VK_PIPELINE_BIND_POINT_COMPUTE
                                                 : VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline.layout(), 0, 1, &set, 0, nullptr);
}

void CommandContext::pushConstants(const Pipeline& pipeline, const void* data,
                                   std::uint32_t bytes) {
    const VkShaderStageFlags stages =
        pipeline.isCompute() ? VK_SHADER_STAGE_COMPUTE_BIT
                             : (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    vkCmdPushConstants(vk(cmd_), pipeline.layout(), stages, 0, bytes, data);
}

void CommandContext::bindVertexBuffer(const Buffer& buffer, std::uint64_t offset) {
    const VkDeviceSize vkOffset = offset;
    const VkBuffer handle = buffer.handle();
    vkCmdBindVertexBuffers(vk(cmd_), 0, 1, &handle, &vkOffset);
}

void CommandContext::bindIndexBuffer(const Buffer& buffer, std::uint64_t offset) {
    vkCmdBindIndexBuffer(vk(cmd_), buffer.handle(), offset, VK_INDEX_TYPE_UINT32);
}

void CommandContext::draw(std::uint32_t vertexCount, std::uint32_t instanceCount,
                          std::uint32_t firstVertex, std::uint32_t firstInstance) {
    vkCmdDraw(vk(cmd_), vertexCount, instanceCount, firstVertex, firstInstance);
}

void CommandContext::drawIndexed(std::uint32_t indexCount, std::uint32_t instanceCount,
                                 std::uint32_t firstIndex, std::int32_t vertexOffset,
                                 std::uint32_t firstInstance) {
    vkCmdDrawIndexed(vk(cmd_), indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

void CommandContext::drawIndirect(const Buffer& buffer, std::uint64_t offset,
                                  std::uint32_t drawCount, std::uint32_t stride) {
    vkCmdDrawIndirect(vk(cmd_), buffer.handle(), offset, drawCount,
                      stride == 0 ? sizeof(VkDrawIndirectCommand) : stride);
}

void CommandContext::drawIndexedIndirect(const Buffer& buffer, std::uint64_t offset,
                                         std::uint32_t drawCount, std::uint32_t stride) {
    vkCmdDrawIndexedIndirect(vk(cmd_), buffer.handle(), offset, drawCount,
                             stride == 0 ? sizeof(VkDrawIndexedIndirectCommand) : stride);
}

void CommandContext::dispatch(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    vkCmdDispatch(vk(cmd_), x, y, z);
}

} // namespace rend::gpu

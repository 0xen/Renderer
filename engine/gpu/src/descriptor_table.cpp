#include "rend/gpu/descriptor_table.h"

#include "rend/gpu/device.h"

#include <volk.h>

#include <array>
#include <format>

namespace rend::gpu {

Result<std::unique_ptr<DescriptorTable>> DescriptorTable::create(const Device& device,
                                                                 std::uint32_t maxTextures) {
    auto table = std::unique_ptr<DescriptorTable>(new DescriptorTable());
    table->device_ = &device;

    std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
    bindings[0] = {.binding = 0,
                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                   .descriptorCount = 1,
                   .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT};
    bindings[1] = {.binding = 1,
                   .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                   .descriptorCount = maxTextures,
                   .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT};
    bindings[2] = {.binding = 2,
                   .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                   .descriptorCount = 1,
                   .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT};
    for (std::uint32_t i = 3; i <= 5; ++i) {
        bindings[i] = {.binding = i,
                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                       .descriptorCount = 1,
                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    }

    // The cull bindings (3-5) are partially bound: only written when the
    // compaction pass is active, and never statically used without it.
    const std::array<VkDescriptorBindingFlags, 6> bindingFlags{
        0,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        0,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT};
    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
    flagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flagsInfo.bindingCount = static_cast<std::uint32_t>(bindingFlags.size());
    flagsInfo.pBindingFlags = bindingFlags.data();

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext = &flagsInfo;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (VkResult r =
            vkCreateDescriptorSetLayout(device.handle(), &layoutInfo, nullptr, &table->layout_);
        r != VK_SUCCESS) {
        return Error{std::format("vkCreateDescriptorSetLayout failed ({})", static_cast<int>(r))};
    }

    const std::array<VkDescriptorPoolSize, 3> poolSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, maxTextures},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 1}};
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    if (VkResult r = vkCreateDescriptorPool(device.handle(), &poolInfo, nullptr, &table->pool_);
        r != VK_SUCCESS) {
        return Error{std::format("vkCreateDescriptorPool failed ({})", static_cast<int>(r))};
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = table->pool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &table->layout_;
    if (VkResult r = vkAllocateDescriptorSets(device.handle(), &allocInfo, &table->set_);
        r != VK_SUCCESS) {
        return Error{std::format("vkAllocateDescriptorSets failed ({})", static_cast<int>(r))};
    }

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
    if (VkResult r = vkCreateSampler(device.handle(), &samplerInfo, nullptr, &table->sampler_);
        r != VK_SUCCESS) {
        return Error{std::format("vkCreateSampler failed ({})", static_cast<int>(r))};
    }

    VkDescriptorImageInfo samplerWrite{};
    samplerWrite.sampler = table->sampler_;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = table->set_;
    write.dstBinding = 2;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    write.pImageInfo = &samplerWrite;
    vkUpdateDescriptorSets(device.handle(), 1, &write, 0, nullptr);

    return table;
}

DescriptorTable::~DescriptorTable() {
    if (!device_) {
        return;
    }
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_->handle(), sampler_, nullptr);
    }
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_->handle(), pool_, nullptr);
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_->handle(), layout_, nullptr);
    }
}

void DescriptorTable::writeObjectBuffer(VkBuffer buffer, std::uint64_t range) {
    VkDescriptorBufferInfo info{.buffer = buffer, .offset = 0, .range = range};
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set_;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &info;
    vkUpdateDescriptorSets(device_->handle(), 1, &write, 0, nullptr);
}

void DescriptorTable::writeStorageBuffer(std::uint32_t binding, VkBuffer buffer,
                                         std::uint64_t range) {
    VkDescriptorBufferInfo info{.buffer = buffer, .offset = 0, .range = range};
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set_;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &info;
    vkUpdateDescriptorSets(device_->handle(), 1, &write, 0, nullptr);
}

void DescriptorTable::writeTexture(std::uint32_t index, VkImageView view) {
    VkDescriptorImageInfo info{};
    info.imageView = view;
    info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set_;
    write.dstBinding = 1;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &info;
    vkUpdateDescriptorSets(device_->handle(), 1, &write, 0, nullptr);
}

} // namespace rend::gpu

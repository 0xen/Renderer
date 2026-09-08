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

    std::vector<VkDescriptorSetLayoutBinding> bindings(10);
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
    // Camera buffer: one region per frame slot; compute-visible for future
    // frustum culling, fragment-visible for traced primary ray generation.
    bindings[6] = {.binding = 6,
                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                   .descriptorCount = 1,
                   .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT |
                                 VK_SHADER_STAGE_FRAGMENT_BIT};
    bindings[7] = {.binding = 7,
                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                   .descriptorCount = 1,
                   .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT};
    bindings[8] = {.binding = 8,
                   .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                   .descriptorCount = 4, // shadow cascades
                   .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT};
    bindings[9] = {.binding = 9,
                   .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                   .descriptorCount = 1,
                   .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT};

    // Bindings 3-8 are partially bound: only written when their passes are
    // active, never statically used without.
    std::vector<VkDescriptorBindingFlags> bindingFlags{
        0,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        0,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        0};

    // The TLAS binding only exists where the extension does — a layout
    // naming an unknown descriptor type is invalid, not "unused".
    const bool rayQuery = device.isEnabled(Feature::RayQuery);
    if (rayQuery) {
        bindings.push_back({.binding = 10,
                            .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                            .descriptorCount = 1,
                            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT});
        bindingFlags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
        // Hit-attribute fetch for traced primary rays: the geometry pool's
        // raw bytes (11) plus per-object firstIndex/vertexOffset (12) let
        // the shader pull the hit triangle's vertices itself.
        for (std::uint32_t binding = 11; binding <= 12; ++binding) {
            bindings.push_back({.binding = binding,
                                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                .descriptorCount = 1,
                                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT});
            bindingFlags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
        }
    }
    // GPU skinning (bindings 13-17, compute): the geometry pool as a
    // writable destination plus the skin/morph/joint inputs. Always in the
    // layout (no extension needed), written only when a scene animates.
    for (std::uint32_t binding = 13; binding <= 17; ++binding) {
        bindings.push_back({.binding = binding,
                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .descriptorCount = 1,
                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT});
        bindingFlags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
    }
    // Per-object world transforms (binding 19, vertex + compute): per-
    // camera-slot regions the CPU rewrites each frame; runtime-spawned
    // models are placed/moved through them, everything else rides
    // identity. The cull pass reads them for per-instance frustum tests.
    bindings.push_back({.binding = 19,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                                      VK_SHADER_STAGE_COMPUTE_BIT});
    bindingFlags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
    // Instance rows (binding 20, vertex): one {objectIndex, transformIndex}
    // pair per drawn instance; SV_InstanceID resolves through it, so one
    // indirect entry with instanceCount N fans out to N transforms while
    // sharing the same object/material row.
    bindings.push_back({.binding = 20,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT});
    bindingFlags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
    // Frustum culling (compute): binding 21 = the frustum-culled compacted
    // draw list the main scene pass draws (the shadow passes keep drawing
    // binding 4's visibility-only list), 22 = per-slot AABBs per draw
    // entry, 23 = the instance-row buffer again (same VkBuffer as binding
    // 20), writable so partially visible draws compact their surviving
    // instances' rows into per-slot scratch regions, 24 = the transparent
    // draw stream the blend pass draws, 25 = the per-entry mesh LOD
    // tables the cull pass picks index ranges from.
    for (std::uint32_t binding = 21; binding <= 25; ++binding) {
        bindings.push_back({.binding = binding,
                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .descriptorCount = 1,
                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT});
        bindingFlags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
    }
    // Occlusion visibility (binding 26): the proxy pass's fragment shader
    // marks entries whose box survived the depth test; the next frame's
    // cull dispatch reads the other slot's region.
    bindings.push_back({.binding = 26,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT |
                                      VK_SHADER_STAGE_COMPUTE_BIT});
    bindingFlags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
    // Reflection probe cubemap (binding 18): written after the load-time
    // capture; shaders only sample it when the light buffer selects the
    // probe tier, so it may stay unwritten (partially bound).
    bindings.push_back({.binding = 18,
                        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT});
    bindingFlags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

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

    std::vector<VkDescriptorPoolSize> poolSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 21},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, maxTextures + 5},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 2}};
    if (rayQuery) {
        poolSizes.push_back({VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1});
    }
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

    // Shadow comparison sampler: linear filtering over the compare result
    // = free 2x2 hardware PCF; clamp-to-border white so geometry outside
    // the shadow map counts as lit.
    VkSamplerCreateInfo shadowInfo{};
    shadowInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    shadowInfo.magFilter = VK_FILTER_LINEAR;
    shadowInfo.minFilter = VK_FILTER_LINEAR;
    shadowInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadowInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadowInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadowInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    shadowInfo.compareEnable = VK_TRUE;
    shadowInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    if (VkResult r = vkCreateSampler(device.handle(), &shadowInfo, nullptr, &table->shadowSampler_);
        r != VK_SUCCESS) {
        return Error{std::format("vkCreateSampler (shadow) failed ({})", static_cast<int>(r))};
    }

    const std::array<VkDescriptorImageInfo, 2> samplerWrites{
        VkDescriptorImageInfo{.sampler = table->sampler_},
        VkDescriptorImageInfo{.sampler = table->shadowSampler_}};
    std::array<VkWriteDescriptorSet, 2> writes{};
    for (std::size_t i = 0; i < writes.size(); ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = table->set_;
        writes[i].dstBinding = i == 0 ? 2 : 9;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes[i].pImageInfo = &samplerWrites[i];
    }
    vkUpdateDescriptorSets(device.handle(), static_cast<std::uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    return table;
}

DescriptorTable::~DescriptorTable() {
    if (!device_) {
        return;
    }
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_->handle(), sampler_, nullptr);
    }
    if (shadowSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_->handle(), shadowSampler_, nullptr);
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

void DescriptorTable::writeShadowMap(std::uint32_t cascade, VkImageView view) {
    VkDescriptorImageInfo info{};
    info.imageView = view;
    info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set_;
    write.dstBinding = 8;
    write.dstArrayElement = cascade;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &info;
    vkUpdateDescriptorSets(device_->handle(), 1, &write, 0, nullptr);
}

void DescriptorTable::writeProbe(VkImageView view) {
    VkDescriptorImageInfo info{};
    info.imageView = view;
    info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set_;
    write.dstBinding = 18;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &info;
    vkUpdateDescriptorSets(device_->handle(), 1, &write, 0, nullptr);
}

void DescriptorTable::writeAccelerationStructure(VkAccelerationStructureKHR tlas) {
    VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
    asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asInfo.accelerationStructureCount = 1;
    asInfo.pAccelerationStructures = &tlas;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.pNext = &asInfo;
    write.dstSet = set_;
    write.dstBinding = 10;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
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

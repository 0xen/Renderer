#include "rend/gpu/buffer.h"

#include "vulkan/vulkan_types.h"

#include "rend/gpu/device.h"
#include "rend/gpu/memory_tracker.h"

#include <volk.h>

#include <array>
#include <format>

namespace rend::gpu {

namespace {

Result<std::uint32_t> findMemoryType(VkPhysicalDevice physical, std::uint32_t typeBits,
                                     VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    return Error{std::format("No memory type matches bits {:#x} with properties {:#x}", typeBits,
                             static_cast<std::uint32_t>(required))};
}

} // namespace

Result<std::unique_ptr<Buffer>> VulkanBuffer::create(const Device& deviceBase,
                                                     const BufferDesc& desc) {
    const VulkanDevice& device = vk(deviceBase);
    if (desc.size == 0) {
        return Error{"Buffer size must be non-zero"};
    }

    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = desc.size;
    info.usage = desc.usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    const std::array<std::uint32_t, 2> families{device.graphicsQueue().familyIndex,
                                                device.transferQueue().familyIndex};
    if (desc.sharedWithTransferQueue && device.hasDedicatedTransfer()) {
        info.sharingMode = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = static_cast<std::uint32_t>(families.size());
        info.pQueueFamilyIndices = families.data();
    }

    VkBuffer handle = VK_NULL_HANDLE;
    if (VkResult r = vkCreateBuffer(device.handle(), &info, nullptr, &handle); r != VK_SUCCESS) {
        return Error{std::format("vkCreateBuffer failed ({})", static_cast<int>(r))};
    }

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device.handle(), handle, &requirements);

    const VkMemoryPropertyFlags properties =
        desc.location == MemoryLocation::DeviceLocal
            ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    auto typeResult =
        findMemoryType(device.physicalDevice(), requirements.memoryTypeBits, properties);
    if (!typeResult) {
        vkDestroyBuffer(device.handle(), handle, nullptr);
        return typeResult.error();
    }

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex = typeResult.value();

    // Device-address usage requires the matching allocation flag.
    VkMemoryAllocateFlagsInfo allocFlags{};
    allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    if (desc.usage & kUsageShaderDeviceAddress) {
        alloc.pNext = &allocFlags;
    }

    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (VkResult r = vkAllocateMemory(device.handle(), &alloc, nullptr, &memory); r != VK_SUCCESS) {
        vkDestroyBuffer(device.handle(), handle, nullptr);
        return Error{std::format("vkAllocateMemory failed ({}) for {} bytes", static_cast<int>(r),
                                 requirements.size)};
    }
    if (VkResult r = vkBindBufferMemory(device.handle(), handle, memory, 0); r != VK_SUCCESS) {
        vkFreeMemory(device.handle(), memory, nullptr);
        vkDestroyBuffer(device.handle(), handle, nullptr);
        return Error{std::format("vkBindBufferMemory failed ({})", static_cast<int>(r))};
    }

    void* mapped = nullptr;
    if (desc.location == MemoryLocation::HostVisible) {
        if (VkResult r = vkMapMemory(device.handle(), memory, 0, VK_WHOLE_SIZE, 0, &mapped);
            r != VK_SUCCESS) {
            vkFreeMemory(device.handle(), memory, nullptr);
            vkDestroyBuffer(device.handle(), handle, nullptr);
            return Error{std::format("vkMapMemory failed ({})", static_cast<int>(r))};
        }
    }

    auto buffer = std::unique_ptr<VulkanBuffer>(new VulkanBuffer());
    buffer->device_ = &device;
    buffer->buffer_ = handle;
    buffer->memory_ = memory;
    buffer->size_ = desc.size;
    buffer->mapped_ = mapped;
    buffer->allocatedBytes_ = requirements.size;
    buffer->trackKind_ = desc.location == MemoryLocation::DeviceLocal
                             ? MemoryTracker::Kind::DeviceBuffer
                             : MemoryTracker::Kind::HostBuffer;
    MemoryTracker::onAlloc(buffer->trackKind_, requirements.size);
    return std::unique_ptr<Buffer>(std::move(buffer));
}

VulkanBuffer::~VulkanBuffer() {
    if (!device_) {
        return;
    }
    if (mapped_) {
        vkUnmapMemory(device_->handle(), memory_);
    }
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_->handle(), buffer_, nullptr);
    }
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_->handle(), memory_, nullptr);
        MemoryTracker::onFree(trackKind_, allocatedBytes_);
    }
}

Result<std::unique_ptr<Buffer>> Buffer::create(const Device& device, const BufferDesc& desc) {
    switch (device.api()) {
    case Api::Vulkan: return VulkanBuffer::create(device, desc);
    case Api::D3D12: break;
    }
    return Error{std::format("{} backend: Buffer not implemented", apiName(device.api()))};
}

} // namespace rend::gpu

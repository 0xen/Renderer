#include "rend/gpu/image.h"

#include "rend/gpu/device.h"

#include <volk.h>

#include <format>

namespace rend::gpu {

Result<std::unique_ptr<Image>> Image::create(const Device& device, const ImageDesc& desc) {
    if (desc.width == 0 || desc.height == 0) {
        return Error{"Image extent must be non-zero"};
    }

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = static_cast<VkFormat>(desc.format);
    info.extent = {desc.width, desc.height, 1};
    info.mipLevels = desc.mipLevels;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = desc.usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = VK_NULL_HANDLE;
    if (VkResult r = vkCreateImage(device.handle(), &info, nullptr, &image); r != VK_SUCCESS) {
        return Error{std::format("vkCreateImage failed ({})", static_cast<int>(r))};
    }

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device.handle(), image, &requirements);

    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(device.physicalDevice(), &memProps);
    std::uint32_t typeIndex = ~0u;
    for (std::uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            typeIndex = i;
            break;
        }
    }
    if (typeIndex == ~0u) {
        vkDestroyImage(device.handle(), image, nullptr);
        return Error{"No device-local memory type for image"};
    }

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex = typeIndex;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (VkResult r = vkAllocateMemory(device.handle(), &alloc, nullptr, &memory); r != VK_SUCCESS) {
        vkDestroyImage(device.handle(), image, nullptr);
        return Error{std::format("vkAllocateMemory failed ({}) for image", static_cast<int>(r))};
    }
    if (VkResult r = vkBindImageMemory(device.handle(), image, memory, 0); r != VK_SUCCESS) {
        vkFreeMemory(device.handle(), memory, nullptr);
        vkDestroyImage(device.handle(), image, nullptr);
        return Error{std::format("vkBindImageMemory failed ({})", static_cast<int>(r))};
    }

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = info.format;
    viewInfo.subresourceRange = {
        desc.depth ? static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT)
                   : static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_COLOR_BIT),
        0, desc.mipLevels, 0, 1};

    VkImageView view = VK_NULL_HANDLE;
    if (VkResult r = vkCreateImageView(device.handle(), &viewInfo, nullptr, &view);
        r != VK_SUCCESS) {
        vkFreeMemory(device.handle(), memory, nullptr);
        vkDestroyImage(device.handle(), image, nullptr);
        return Error{std::format("vkCreateImageView failed ({})", static_cast<int>(r))};
    }

    auto out = std::unique_ptr<Image>(new Image());
    out->device_ = &device;
    out->image_ = image;
    out->memory_ = memory;
    out->view_ = view;
    out->format_ = desc.format;
    out->width_ = desc.width;
    out->height_ = desc.height;
    out->mipLevels_ = desc.mipLevels;
    return out;
}

Image::~Image() {
    if (!device_) {
        return;
    }
    if (view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_->handle(), view_, nullptr);
    }
    if (image_ != VK_NULL_HANDLE) {
        vkDestroyImage(device_->handle(), image_, nullptr);
    }
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_->handle(), memory_, nullptr);
    }
}

} // namespace rend::gpu

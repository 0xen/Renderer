#include "rend/gpu/swapchain.h"

#include "rend/core/log.h"
#include "rend/gpu/device.h"
#include "rend/gpu/instance.h"

#include <volk.h>

#include <algorithm>

namespace rend::gpu {

namespace {

VkSurfaceFormatKHR chooseFormat(VkPhysicalDevice pd, VkSurfaceKHR surface) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &count, formats.data());

    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) {
            return f;
        }
    }
    return formats.at(0);
}

VkPresentModeKHR choosePresentMode(VkPhysicalDevice pd, VkSurfaceKHR surface, bool vsync) {
    if (vsync) {
        return VK_PRESENT_MODE_FIFO_KHR; // always available
    }
    std::uint32_t count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &count, nullptr);
    std::vector<VkPresentModeKHR> modes(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &count, modes.data());
    for (VkPresentModeKHR m : modes) {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) {
            return m;
        }
    }
    for (VkPresentModeKHR m : modes) {
        if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) {
            log::warn("No MAILBOX present mode; using IMMEDIATE (expect tearing)");
            return m;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkCompositeAlphaFlagBitsKHR chooseCompositeAlpha(const VkSurfaceCapabilitiesKHR& caps, bool transparent) {
    if (transparent) {
        if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) {
            return VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
        }
        if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) {
            return VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
        }
        log::warn("Transparent target requested but driver offers no alpha compositing; falling back to "
                  "opaque");
    }
    if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) {
        return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    }
    return VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
}

} // namespace

Result<std::unique_ptr<Swapchain>> Swapchain::create(const Instance& instance, const Device& device,
                                                     const SwapchainDesc& desc) {
    VkBool32 presentable = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(device.physicalDevice(), device.graphicsQueue().familyIndex,
                                         desc.surface, &presentable);
    if (!presentable) {
        return Error{"Graphics queue family cannot present to this surface"};
    }

    auto swapchain = std::unique_ptr<Swapchain>(new Swapchain());
    swapchain->instance_ = &instance;
    swapchain->device_ = &device;
    swapchain->surface_ = desc.surface;
    swapchain->transparent_ = desc.transparent;
    swapchain->vsync_ = desc.vsync;

    if (auto r = swapchain->build(desc.width, desc.height, VK_NULL_HANDLE); !r) {
        return r.error();
    }
    return swapchain;
}

Result<void> Swapchain::build(std::uint32_t width, std::uint32_t height, VkSwapchainKHR old) {
    VkPhysicalDevice pd = device_->physicalDevice();

    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface_, &caps);

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == 0xFFFFFFFFu) { // surface lets us choose
        extent.width = std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    if (extent.width == 0 || extent.height == 0) {
        return Error{"Surface has zero extent (minimized?)"};
    }

    std::uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) {
        imageCount = std::min(imageCount, caps.maxImageCount);
    }

    const VkSurfaceFormatKHR format = chooseFormat(pd, surface_);
    const VkPresentModeKHR presentMode = choosePresentMode(pd, surface_, vsync_);
    const VkCompositeAlphaFlagBitsKHR alpha = chooseCompositeAlpha(caps, transparent_);

    VkSwapchainCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface = surface_;
    info.minImageCount = imageCount;
    info.imageFormat = format.format;
    info.imageColorSpace = format.colorSpace;
    info.imageExtent = extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = caps.currentTransform;
    info.compositeAlpha = alpha;
    info.presentMode = presentMode;
    info.clipped = VK_TRUE;
    info.oldSwapchain = old;

    VkSwapchainKHR handle = VK_NULL_HANDLE;
    if (VkResult r = vkCreateSwapchainKHR(device_->handle(), &info, nullptr, &handle); r != VK_SUCCESS) {
        return Error{std::format("vkCreateSwapchainKHR failed ({})", static_cast<int>(r))};
    }
    if (old != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_->handle(), old, nullptr);
    }
    swapchain_ = handle;
    format_ = format.format;
    width_ = extent.width;
    height_ = extent.height;

    std::uint32_t actualCount = 0;
    vkGetSwapchainImagesKHR(device_->handle(), swapchain_, &actualCount, nullptr);
    images_.resize(actualCount);
    vkGetSwapchainImagesKHR(device_->handle(), swapchain_, &actualCount, images_.data());

    views_.resize(actualCount);
    for (std::uint32_t i = 0; i < actualCount; ++i) {
        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = images_[i];
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = format.format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (VkResult r = vkCreateImageView(device_->handle(), &view, nullptr, &views_[i]);
            r != VK_SUCCESS) {
            return Error{std::format("vkCreateImageView failed ({})", static_cast<int>(r))};
        }
    }

    log::info("Swapchain {}x{}: {} images, format {}, present mode {}, composite alpha {}", width_,
              height_, actualCount, static_cast<int>(format.format), static_cast<int>(presentMode),
              static_cast<int>(alpha));
    return {};
}

Result<void> Swapchain::recreate(std::uint32_t width, std::uint32_t height) {
    vkDeviceWaitIdle(device_->handle());
    destroyViews();
    VkSwapchainKHR old = swapchain_;
    swapchain_ = VK_NULL_HANDLE;
    return build(width, height, old);
}

void Swapchain::destroyViews() {
    for (VkImageView view : views_) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(device_->handle(), view, nullptr);
        }
    }
    views_.clear();
    images_.clear();
}

Swapchain::~Swapchain() {
    if (device_ && device_->handle()) {
        vkDeviceWaitIdle(device_->handle());
        destroyViews();
        if (swapchain_ != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device_->handle(), swapchain_, nullptr);
        }
    }
    if (surface_ != VK_NULL_HANDLE && instance_) {
        vkDestroySurfaceKHR(instance_->handle(), surface_, nullptr);
        log::info("Swapchain and surface destroyed");
    }
}

} // namespace rend::gpu

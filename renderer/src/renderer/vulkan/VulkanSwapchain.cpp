#include <renderer/vulkan/VulkanSwapchain.hpp>
#include <renderer/vulkan/VulkanPhysicalDevice.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanInstance.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>
#include <renderer/vulkan/VulkanGraphicsPipeline.hpp>
#include <renderer/vulkan/VulkanRaytracePipeline.hpp>
#include <renderer/vulkan/VulkanRenderer.hpp>
#include <renderer/vulkan/VulkanDescriptorPool.hpp>
#include <renderer/vulkan/VulkanDescriptorSet.hpp>

#include <assert.h>

Renderer::Vulkan::VulkanSwapchain::VulkanSwapchain(VulkanRenderer* renderer, VulkanInstance * instance, VulkanDevice * device, VkSurfaceKHR* surface, Renderer::NativeWindowHandle* window_handle)
{
	m_renderer = renderer;
	m_instance = instance;
	m_device = device;
	m_surface = surface;
	m_window_handle = window_handle;

	CreateSwapchain();
	InitSemaphores();
	m_wait_stages = new VkPipelineStageFlags{ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

	
}

Renderer::Vulkan::VulkanSwapchain::~VulkanSwapchain()
{
	DeInitSemaphores();
	DestroySwapchain();
	delete m_wait_stages;
}

void Renderer::Vulkan::VulkanSwapchain::RebuildSwapchain()
{
	vkDeviceWaitIdle(*m_device->GetVulkanDevice());
	DestroySwapchain();

	CreateSwapchain();
}


VkImageView Renderer::Vulkan::VulkanSwapchain::GetBackBufferImage(unsigned int index)
{
	return m_swap_chain_image_views[index];
}

VkDescriptorImageInfo Renderer::Vulkan::VulkanSwapchain::GetRayTraceStagingBuffer()
{
	return VulkanInitializers::DescriptorImageInfo(nullptr, m_raytrace_storage_image_view, VK_IMAGE_LAYOUT_GENERAL);
}

VkSubmitInfo Renderer::Vulkan::VulkanSwapchain::GetSubmitInfo()
{
	VkSubmitInfo sumbit_info = {};
	sumbit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	sumbit_info.waitSemaphoreCount = 1;
	sumbit_info.pWaitSemaphores = &m_image_available_semaphore;
	sumbit_info.pWaitDstStageMask = m_wait_stages;
	sumbit_info.commandBufferCount = 1;
	sumbit_info.signalSemaphoreCount = 1;
	sumbit_info.pSignalSemaphores = &m_render_finished_semaphore;
	return sumbit_info;
}

void Renderer::Vulkan::VulkanSwapchain::Present(std::vector<VkSemaphore> signal_semaphores, unsigned int frame_index)
{
	VkResult present_result = VkResult::VK_RESULT_MAX_ENUM;
	VkSwapchainKHR swap_chains[] = { m_swap_chain };
	VkPresentInfoKHR present_info = {};
	present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present_info.waitSemaphoreCount = (uint32_t)signal_semaphores.size();
	present_info.pWaitSemaphores = signal_semaphores.data();
	present_info.swapchainCount = 1;
	present_info.pSwapchains = swap_chains;
	present_info.pImageIndices = &frame_index;
	present_info.pResults = nullptr;

	vkQueuePresentKHR(
		*m_device->GetPresentQueue(),
		&present_info
	);
	vkDeviceWaitIdle(*m_device->GetVulkanDevice());
	//m_frame_index = (m_frame_index + 1) % m_swap_chain_images.size();
}

VkSurfaceKHR * Renderer::Vulkan::VulkanSwapchain::GetSurface()
{
	return m_surface;
}

VkSwapchainKHR Renderer::Vulkan::VulkanSwapchain::GetSwapchain()
{
	return m_swap_chain;
}

VkSurfaceFormatKHR Renderer::Vulkan::VulkanSwapchain::GetSurfaceFormat()
{
	return surface_format;
}

VkPresentModeKHR Renderer::Vulkan::VulkanSwapchain::GetSurfacePresentMode()
{
	return present_mode;
}
Renderer::NativeWindowHandle * Renderer::Vulkan::VulkanSwapchain::GetNativeWindowHandle()
{
	return m_window_handle;
}

uint32_t Renderer::Vulkan::VulkanSwapchain::GetImageCount()
{
	return image_count;
}

VkImage Renderer::Vulkan::VulkanSwapchain::GetRaytracingScratchImage()
{
	return m_raytrace_storage_image;
}

std::vector<VkImage> Renderer::Vulkan::VulkanSwapchain::GetSwapchainImages()
{
	return m_swap_chain_images;
}

std::vector<VkImageView> Renderer::Vulkan::VulkanSwapchain::GetSwpachainImageViews()
{
	return m_swap_chain_image_views;
}

VkSemaphore Renderer::Vulkan::VulkanSwapchain::GetImageAvailableSemaphore()
{
	return m_image_available_semaphore;
}

VkSemaphore Renderer::Vulkan::VulkanSwapchain::GetRenderFinishedSemaphore()
{
	return m_render_finished_semaphore;
}

VkFormat Renderer::Vulkan::VulkanSwapchain::GetSwapChainImageFormat()
{
	return m_swap_chain_image_format;
}

VkExtent2D Renderer::Vulkan::VulkanSwapchain::GetSwapchainExtent()
{
	return m_swap_chain_extent;
}

void Renderer::Vulkan::VulkanSwapchain::CreateSwapchain()
{
	InitSwapchain();
	InitSwapchainImages();
	if (m_instance->GetFlags()&VulkanFlags::Raytrace == VulkanFlags::Raytrace)
	{
		InitRaytracingTempImage();
	}
}

void Renderer::Vulkan::VulkanSwapchain::DestroySwapchain()
{
	if (m_instance->GetFlags()&VulkanFlags::Raytrace == VulkanFlags::Raytrace)
	{
		DeInitRaytracingTempImage();
	}
	DeInitSwapchainImages();
	DeInitSwapchain();
}

void Renderer::Vulkan::VulkanSwapchain::InitSwapchain()
{
	// Get all required information for the swapchain initialization
	VulkanSwapChainSupport swap_chain_support;
	CheckSwapChainSupport(swap_chain_support);

	surface_format = ChooseSwapSurfaceFormat(swap_chain_support.formats);
	present_mode = ChooseSwapPresentMode(swap_chain_support.present_modes);
	VkExtent2D extent = ChooseSwapExtent(swap_chain_support.capabilities);

	// Store the format and extent for later use
	m_swap_chain_extent = extent;
	m_swap_chain_image_format = surface_format.format;

	// Make sure that the amount of images we are creating for the swapchain are in between the min and max
	image_count = swap_chain_support.capabilities.minImageCount + 1;
	if (swap_chain_support.capabilities.maxImageCount > 0 && image_count > swap_chain_support.capabilities.maxImageCount)
	{
		image_count = swap_chain_support.capabilities.maxImageCount;
	}

	VulkanQueueFamilyIndices indices = *m_device->GetVulkanPhysicalDevice()->GetQueueFamilies();
	VkSwapchainCreateInfoKHR create_info = VulkanInitializers::SwapchainCreateInfoKHR(surface_format, extent, present_mode, image_count, *m_surface, indices, swap_chain_support);

	if ((swap_chain_support.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT) == VK_IMAGE_USAGE_STORAGE_BIT &&
		(m_instance->GetFlags() & Renderer::Vulkan::VulkanFlags::Raytrace) == Renderer::Vulkan::VulkanFlags::Raytrace)
	{
		create_info.imageUsage |= VK_IMAGE_USAGE_STORAGE_BIT;
	}

	if ((swap_chain_support.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == VK_IMAGE_USAGE_TRANSFER_DST_BIT)
	{
		create_info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	}

	if ((swap_chain_support.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
	{
		create_info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	}



	ErrorCheck(vkCreateSwapchainKHR(
		*m_device->GetVulkanDevice(),
		&create_info,
		nullptr,
		&m_swap_chain
	));

	assert(!HasError() && "Unable to create a swapchain");

	// Get the swapchain image count
	ErrorCheck(vkGetSwapchainImagesKHR(
		*m_device->GetVulkanDevice(),
		m_swap_chain,
		&image_count,
		nullptr
	));
	assert(!HasError() && "Unable to get swapchain images");
	m_swap_chain_images.resize(image_count);

	// Get the swapchain images
	ErrorCheck(vkGetSwapchainImagesKHR(
		*m_device->GetVulkanDevice(),
		m_swap_chain,
		&image_count,
		m_swap_chain_images.data()
	));



	assert(!HasError() && "Unable to get swapchain images");
}

void Renderer::Vulkan::VulkanSwapchain::DeInitSwapchain()
{
	vkDestroySwapchainKHR(
		*m_device->GetVulkanDevice(),
		m_swap_chain,
		nullptr
	);
}

void Renderer::Vulkan::VulkanSwapchain::CheckSwapChainSupport(VulkanSwapChainSupport & support)
{
	// Get the surface capabilities
	ErrorCheck(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
		*m_device->GetVulkanPhysicalDevice()->GetPhysicalDevice(),
		*m_surface,
		&support.capabilities
	));

	assert(!HasError() && "Unable to get surface capabilities");

	// Get all formats
	uint32_t format_count = 0;
	ErrorCheck(vkGetPhysicalDeviceSurfaceFormatsKHR(*m_device->GetVulkanPhysicalDevice()->GetPhysicalDevice(), *m_surface, &format_count, nullptr));

	assert(!HasError() && "Unable to get surface capabilities");
	if (format_count == 0) return;
	support.formats.resize(format_count);
	ErrorCheck(vkGetPhysicalDeviceSurfaceFormatsKHR(*m_device->GetVulkanPhysicalDevice()->GetPhysicalDevice(), *m_surface, &format_count, support.formats.data()));

	assert(!HasError() && "Unable to get surface capabilities");
	// Get all the present modes
	uint32_t present_mode_count;
	ErrorCheck(vkGetPhysicalDeviceSurfacePresentModesKHR(*m_device->GetVulkanPhysicalDevice()->GetPhysicalDevice(), *m_surface, &present_mode_count, nullptr));

	assert(!HasError() && "Unable to get surface capabilities");
	if (present_mode_count == 0) return;
	support.present_modes.resize(present_mode_count);
	ErrorCheck(vkGetPhysicalDeviceSurfacePresentModesKHR(*m_device->GetVulkanPhysicalDevice()->GetPhysicalDevice(), *m_surface, &present_mode_count, support.present_modes.data()));

	assert(!HasError() && "Unable to get surface capabilities");
}

VkSurfaceFormatKHR Renderer::Vulkan::VulkanSwapchain::ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available_formats)
{
	// If there is one format and that format is VK_FORMAT_UNDEFINED, the vulkan dose not mind what we set as a format
	if (available_formats.size() == 1 && available_formats[0].format == VK_FORMAT_UNDEFINED)
	{
		return{ VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
	}
	// Loop through formats and find the most appropriate one
	for (const auto& available_format : available_formats)
	{
		if (available_format.format == VK_FORMAT_B8G8R8A8_UNORM && available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
		{
			return available_format;
		}
	}
	// If we have got this far, its best just to go with the format specified
	if (available_formats.size() > 0)return available_formats[0];

	return{ VK_FORMAT_UNDEFINED };
}

VkPresentModeKHR Renderer::Vulkan::VulkanSwapchain::ChooseSwapPresentMode(const std::vector<VkPresentModeKHR> available_present_modes)
{
	// Loop through modes
	for (const auto& available_present_mode : available_present_modes)
	{
		// Loop for tipple buffering
		if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR)
		{
			// Temporerply dissable everything but V-SYNC
			return available_present_mode;
		}
	}
	// Fallback format that is guaranteed to be available
	// This locks the rendering to V-Sync
	return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D Renderer::Vulkan::VulkanSwapchain::ChooseSwapExtent(const VkSurfaceCapabilitiesKHR & capabilities)
{
	if (capabilities.currentExtent.width != UINT32_MAX)
	{
		return capabilities.currentExtent;
	}
	else
	{
		VkExtent2D extent = { (uint32_t)m_window_handle->width, (uint32_t)m_window_handle->height };
		// Make sure the swap buffer image size is not too big or too small
		if (extent.width > capabilities.maxImageExtent.width)extent.width = capabilities.maxImageExtent.width;
		if (extent.width < capabilities.minImageExtent.width)extent.width = capabilities.minImageExtent.width;
		if (extent.height > capabilities.maxImageExtent.width)extent.height = capabilities.maxImageExtent.height;
		if (extent.height < capabilities.minImageExtent.width)extent.height = capabilities.minImageExtent.height;
		return extent;
	}
}

void Renderer::Vulkan::VulkanSwapchain::InitSwapchainImages()
{
	// Resize the new array
	m_swap_chain_image_views.resize(m_swap_chain_images.size());
	int i = 0;
	for (auto swapchain_image : m_swap_chain_images)
	{
		VulkanCommon::CreateImageView(m_device, swapchain_image, m_swap_chain_image_format, VK_IMAGE_ASPECT_COLOR_BIT, m_swap_chain_image_views[i]);
		i++;
	}
}

void Renderer::Vulkan::VulkanSwapchain::DeInitSwapchainImages()
{
	for (uint32_t i = 0; i < m_swap_chain_image_views.size(); i++)
	{
		vkDestroyImageView(
			*m_device->GetVulkanDevice(),
			m_swap_chain_image_views[i],
			nullptr
		);
	}
	m_swap_chain_image_views.clear();
}


void Renderer::Vulkan::VulkanSwapchain::InitRaytracingTempImage()
{
	VulkanCommon::CreateImage(m_device, m_swap_chain_extent, m_swap_chain_image_format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_raytrace_storage_image, m_raytrace_storage_image_memory);
	VulkanCommon::CreateImageView(m_device, m_raytrace_storage_image, m_swap_chain_image_format, VK_IMAGE_ASPECT_COLOR_BIT, m_raytrace_storage_image_view);

	VulkanCommon::TransitionImageLayout(m_device, m_raytrace_storage_image, m_swap_chain_image_format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });
}

void Renderer::Vulkan::VulkanSwapchain::DeInitRaytracingTempImage()
{
	vkDestroyImageView(
		*m_device->GetVulkanDevice(),
		m_raytrace_storage_image_view,
		nullptr
	);
	vkFreeMemory(
		*m_device->GetVulkanDevice(),
		m_raytrace_storage_image_memory,
		nullptr
	);
	vkDestroyImage(
		*m_device->GetVulkanDevice(),
		m_raytrace_storage_image,
		nullptr
	);
}

void Renderer::Vulkan::VulkanSwapchain::InitSemaphores()
{
	VkSemaphoreCreateInfo semaphore_info = VulkanInitializers::SemaphoreCreateInfo();

	ErrorCheck(vkCreateSemaphore(*m_device->GetVulkanDevice(), &semaphore_info, nullptr, &m_image_available_semaphore));
	assert(!HasError() && "Unable to create semaphore");

	ErrorCheck(vkCreateSemaphore(*m_device->GetVulkanDevice(), &semaphore_info, nullptr, &m_render_finished_semaphore));
	assert(!HasError() && "Unable to create semaphore");
}

void Renderer::Vulkan::VulkanSwapchain::DeInitSemaphores()
{
	if (m_image_available_semaphore != VK_NULL_HANDLE)
	{
		vkDestroySemaphore(*m_device->GetVulkanDevice(), m_image_available_semaphore, nullptr);
		m_image_available_semaphore = VK_NULL_HANDLE;
	}
	if (m_render_finished_semaphore != VK_NULL_HANDLE)
	{
		vkDestroySemaphore(*m_device->GetVulkanDevice(), m_render_finished_semaphore, nullptr);
		m_render_finished_semaphore = VK_NULL_HANDLE;
	}
}

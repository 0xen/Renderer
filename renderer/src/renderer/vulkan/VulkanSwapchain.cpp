#include <renderer/vulkan/VulkanSwapchain.hpp>
#include <renderer/vulkan/VulkanPhysicalDevice.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanInstance.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>
#include <renderer/vulkan/VulkanGraphicsPipeline.hpp>

#include <assert.h>

Renderer::Vulkan::VulkanSwapchain::VulkanSwapchain(VulkanInstance * instance, VulkanDevice * device, VkSurfaceKHR* surface, Renderer::NativeWindowHandle* window_handle)
{
	m_instance = instance;
	m_device = device;
	m_surface = surface;
	m_window_handle = window_handle;

	CreateSwapchain();
	InitCommandBuffers();
	InitSemaphores();
	m_should_rebuild_cmd = true;
	m_wait_stages = new VkPipelineStageFlags{ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
}

Renderer::Vulkan::VulkanSwapchain::~VulkanSwapchain()
{
	DeInitSemaphores();
	DestroySwapchain();
	delete m_wait_stages;
}

void Renderer::Vulkan::VulkanSwapchain::RequestRebuildCommandBuffers()
{
	m_should_rebuild_cmd = true;
}

void Renderer::Vulkan::VulkanSwapchain::RebuildSwapchain()
{
	vkDeviceWaitIdle(*m_device->GetVulkanDevice());
	DestroySwapchain();

	CreateSwapchain();

	// Build initial command buffers
	RebuildCommandBuffers();
}

unsigned int Renderer::Vulkan::VulkanSwapchain::GetCurrentBuffer()
{
	if (m_should_rebuild_cmd)
	{
		m_should_rebuild_cmd = false;
		RebuildCommandBuffers();
	}
	for (auto pipeline : m_pipelines)
	{
		if (pipeline->HasChanged())
		{
			RebuildCommandBuffers();
		}
	}
	VkResult check = vkAcquireNextImageKHR(
		*m_device->GetVulkanDevice(),
		m_swap_chain,
		UINT32_MAX,
		m_image_available_semaphore,
		VK_NULL_HANDLE,
		&m_active_swapchain_image
	);
	if (check == VK_ERROR_OUT_OF_DATE_KHR)
	{
		RebuildSwapchain();
		return GetCurrentBuffer();
	}
	ErrorCheck(check);
	assert(!HasError());
	vkQueueWaitIdle(
		*m_device->GetPresentQueue()
	);
	return m_active_swapchain_image;
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

void Renderer::Vulkan::VulkanSwapchain::SubmitQueue(unsigned int currentBuffer)
{
	VkSubmitInfo info = GetSubmitInfo();
	info.pCommandBuffers = &m_command_buffers[currentBuffer];
	ErrorCheck(vkQueueSubmit(
		*m_device->GetGraphicsQueue(),
		1,
		&info,
		VK_NULL_HANDLE
	));

	assert(!HasError());
}

void Renderer::Vulkan::VulkanSwapchain::Present(std::vector<VkSemaphore> signal_semaphores)
{
	VkResult present_result = VkResult::VK_RESULT_MAX_ENUM;
	VkSwapchainKHR swap_chains[] = { m_swap_chain };
	VkPresentInfoKHR present_info = {};
	present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present_info.waitSemaphoreCount = signal_semaphores.size();
	present_info.pWaitSemaphores = signal_semaphores.data();
	present_info.swapchainCount = 1;
	present_info.pSwapchains = swap_chains;
	present_info.pImageIndices = &m_active_swapchain_image;
	present_info.pResults = nullptr;

	vkQueuePresentKHR(
		*m_device->GetPresentQueue(),
		&present_info
	);
	vkDeviceWaitIdle(*m_device->GetVulkanDevice());
}

VkRenderPass * Renderer::Vulkan::VulkanSwapchain::GetRenderPass()
{
	return &m_render_pass;
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

void Renderer::Vulkan::VulkanSwapchain::AttachGraphicsPipeline(VulkanGraphicsPipeline * pipeline, bool priority)
{
	if (priority)
	{
		m_pipelines.push_back(pipeline);
	}
	else
	{
		m_pipelines.insert(m_pipelines.begin(), pipeline);
	}
}

void Renderer::Vulkan::VulkanSwapchain::RemoveGraphicsPipeline(VulkanGraphicsPipeline * pipeline)
{
	auto it = std::find(m_pipelines.begin(), m_pipelines.end(), pipeline);
	if (it != m_pipelines.end())
	{
		m_pipelines.erase(it);
		RebuildSwapchain();
	}
}

uint32_t Renderer::Vulkan::VulkanSwapchain::GetImageCount()
{
	return image_count;
}

std::vector<VkImage> Renderer::Vulkan::VulkanSwapchain::GetSwapchainImages()
{
	return m_swap_chain_images;
}

std::vector<VkImageView> Renderer::Vulkan::VulkanSwapchain::GetSwpachainImageViews()
{
	return m_swap_chain_image_views;
}

std::vector<VkFramebuffer> Renderer::Vulkan::VulkanSwapchain::GetSwapchainFrameBuffers()
{
	return m_swap_chain_framebuffers;
}

std::vector<VkCommandBuffer> Renderer::Vulkan::VulkanSwapchain::GetCommandBuffers()
{
	return m_command_buffers;
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

VkImage Renderer::Vulkan::VulkanSwapchain::GetDepthImage()
{
	return m_depth_image;
}

VkExtent2D Renderer::Vulkan::VulkanSwapchain::GetSwapchainExtent()
{
	return m_swap_chain_extent;
}

void Renderer::Vulkan::VulkanSwapchain::RebuildCommandBuffers()
{
	VkCommandBufferBeginInfo begin_info = VulkanInitializers::CommandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
	std::array<VkClearValue, 2> clear_values;
	
	std::copy(std::begin(m_window_handle->clear_color.float32), std::end(m_window_handle->clear_color.float32), std::begin(clear_values[0].color.float32));
	clear_values[1].depthStencil = { 1.0f, 0 };

	VkRenderPassBeginInfo render_pass_info = VulkanInitializers::RenderPassBeginInfo(m_render_pass, m_swap_chain_extent, clear_values);

	for (uint32_t i = 0; i < m_command_buffers.size(); i++)
	{
		// Reset the command buffers
		vkResetCommandBuffer(
			m_command_buffers[i],
			0
		);
		// Setup unique frame buffer
		render_pass_info.framebuffer = m_swap_chain_framebuffers[i];


		ErrorCheck(vkBeginCommandBuffer(
			m_command_buffers[i],
			&begin_info
		));

		assert(!HasError() && "Unable to create command buffer");

		vkCmdBeginRenderPass(
			m_command_buffers[i],
			&render_pass_info,
			VK_SUBPASS_CONTENTS_INLINE
		);


		vkCmdSetLineWidth(m_command_buffers[i], 1.0f);
		const VkViewport viewport = VulkanInitializers::Viewport(m_window_handle->width, m_window_handle->height, 0, 0, 0.0f, 1.0f);
		const VkRect2D scissor = VulkanInitializers::Scissor(m_window_handle->width, m_window_handle->height);
		vkCmdSetViewport(m_command_buffers[i], 0, 1, &viewport);
		vkCmdSetScissor(m_command_buffers[i], 0, 1, &scissor);

		for (auto pipeline : m_pipelines)
		{
			pipeline->AttachToCommandBuffer(m_command_buffers[i]);
		}

		vkCmdEndRenderPass(
			m_command_buffers[i]
		);

		ErrorCheck(vkEndCommandBuffer(
			m_command_buffers[i]
		));

		assert(!HasError() && "Unable to end command buffer");

	}
}

void Renderer::Vulkan::VulkanSwapchain::CreateSwapchain()
{
	InitSwapchain();
	InitSwapchainImages();
	InitRenderPass();
	InitDepthImage();
	InitFrameBuffer();
}

void Renderer::Vulkan::VulkanSwapchain::DestroySwapchain()
{
	DeInitFrameBuffer();
	DeInitDepthImage();
	DeInitRenderPass();
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

void Renderer::Vulkan::VulkanSwapchain::InitRenderPass()
{
	std::vector<VkAttachmentDescription> attachments = {
		VulkanInitializers::AttachmentDescription(m_swap_chain_image_format, VK_ATTACHMENT_STORE_OP_STORE,VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),	//Color
		VulkanInitializers::AttachmentDescription(VulkanCommon::GetDepthImageFormat(m_device), VK_ATTACHMENT_STORE_OP_DONT_CARE,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)		// Depth
	};

	VkAttachmentReference color_attachment_refrence = VulkanInitializers::AttachmentReference(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0);
	VkAttachmentReference depth_attachment_refrence = VulkanInitializers::AttachmentReference(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);

	VkSubpassDescription subpass = VulkanInitializers::SubpassDescription(color_attachment_refrence, depth_attachment_refrence);

	VkSubpassDependency subpass_dependency = VulkanInitializers::SubpassDependency();

	VkRenderPassCreateInfo render_pass_info = VulkanInitializers::RenderPassCreateInfo(attachments, subpass, subpass_dependency);

	ErrorCheck(vkCreateRenderPass(
		*m_device->GetVulkanDevice(),
		&render_pass_info,
		nullptr,
		&m_render_pass
	));
	assert(!HasError() && "Unable to initialize render pass");
}

void Renderer::Vulkan::VulkanSwapchain::DeInitRenderPass()
{
	vkDestroyRenderPass(
		*m_device->GetVulkanDevice(),
		m_render_pass,
		nullptr);
}

void Renderer::Vulkan::VulkanSwapchain::InitCommandBuffers()
{
	// Resize the command buffers so that there is one for each frame buffer
	m_command_buffers.resize(m_swap_chain_framebuffers.size());
	// Setup the allocation info
	VkCommandBufferAllocateInfo alloc_info = VulkanInitializers::CommandBufferAllocateInfo(
		*m_device->GetGraphicsCommandPool(),
		static_cast<uint32_t>(m_command_buffers.size())
	);
	// Allocate the buffers
	ErrorCheck(vkAllocateCommandBuffers(
		*m_device->GetVulkanDevice(),
		&alloc_info,
		m_command_buffers.data()
	));
	assert(!HasError() && "Unable to allocate command buffers");
}

void Renderer::Vulkan::VulkanSwapchain::InitDepthImage()
{
	m_depth_image_format = VulkanCommon::GetDepthImageFormat(m_device);
	VulkanCommon::CreateImage(m_device,m_swap_chain_extent, m_depth_image_format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_depth_image, m_depth_image_memory);
	VulkanCommon::CreateImageView(m_device,m_depth_image, m_depth_image_format, VK_IMAGE_ASPECT_DEPTH_BIT, m_depth_image_view);

	//VulkanCommon::TransitionImageLayout(m_device, m_depth_image, m_depth_image_format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}

void Renderer::Vulkan::VulkanSwapchain::DeInitDepthImage()
{
	vkDestroyImageView(
		*m_device->GetVulkanDevice(),
		m_depth_image_view,
		nullptr
	);
	vkFreeMemory(
		*m_device->GetVulkanDevice(),
		m_depth_image_memory,
		nullptr
	);
	vkDestroyImage(
		*m_device->GetVulkanDevice(),
		m_depth_image,
		nullptr
	);
}

void Renderer::Vulkan::VulkanSwapchain::InitFrameBuffer()
{
	m_swap_chain_framebuffers.resize(m_swap_chain_image_views.size());
	for (uint32_t i = 0; i < m_swap_chain_image_views.size(); i++)
	{
		std::vector<VkImageView> attachments = {
			m_swap_chain_image_views[i],
			m_depth_image_view
		};
		// Frame buffer create info
		VkFramebufferCreateInfo framebuffer_info = VulkanInitializers::FramebufferCreateInfo(m_swap_chain_extent, attachments, m_render_pass);

		ErrorCheck(vkCreateFramebuffer(
			*m_device->GetVulkanDevice(),
			&framebuffer_info,
			nullptr,
			&m_swap_chain_framebuffers[i]
		));
		assert(!HasError() && "Unable to create frame buffer");
	}
}

void Renderer::Vulkan::VulkanSwapchain::DeInitFrameBuffer()
{
	for (auto framebuffer : m_swap_chain_framebuffers)
	{
		vkDestroyFramebuffer(
			*m_device->GetVulkanDevice(),
			framebuffer,
			nullptr
		);
	}
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
	vkDestroySemaphore(*m_device->GetVulkanDevice(), m_image_available_semaphore, nullptr);
	vkDestroySemaphore(*m_device->GetVulkanDevice(), m_render_finished_semaphore, nullptr);
}

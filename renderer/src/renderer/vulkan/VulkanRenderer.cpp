#include <renderer/vulkan/VulkanRenderer.hpp>
#include <renderer/vulkan/VulkanInstance.hpp>
#include <renderer/vulkan/VulkanPhysicalDevice.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanSwapchain.hpp>
#include <renderer\vulkan\VulkanUniformBuffer.hpp>
#include <renderer\vulkan\VulkanVertexBuffer.hpp>
#include <renderer\vulkan\VulkanIndexBuffer.hpp>
#include <renderer\vulkan\VulkanComputePipeline.hpp>
#include <renderer\vulkan\VulkanGraphicsPipeline.hpp>
#include <renderer\vulkan\VulkanComputeProgram.hpp>
#include <renderer\vulkan\VulkanModelPool.hpp>
#include <renderer\vulkan\VulkanTextureBuffer.hpp>
#include <renderer\vulkan\VulkanDescriptor.hpp>
#include <renderer\vulkan\VulkanDescriptorPool.hpp>


#include <assert.h>

using namespace Renderer;
using namespace Renderer::Vulkan;
VulkanRenderer::VulkanRenderer() : IRenderer()
{
}

VulkanRenderer::~VulkanRenderer()
{
	Stop();
}

bool VulkanRenderer::Start(Renderer::NativeWindowHandle* window_handle)
{
	m_window_handle = window_handle;

	m_instance = new VulkanInstance();
	Status::ErrorCheck(m_instance);
	if (HasError())return false;

	CreateSurface(window_handle);

	m_physical_device = VulkanPhysicalDevice::GetPhysicalDevice(m_instance, m_surface);
	Status::ErrorCheck(m_physical_device);
	if (HasError())return false;

	m_device = new VulkanDevice(m_instance, m_physical_device);
	Status::ErrorCheck(m_device);
	if (HasError())return false;

	m_swapchain = new VulkanSwapchain(m_instance, m_device, &m_surface, window_handle);
	Status::ErrorCheck(m_swapchain);
	if (HasError())return false;


	return true;
}

void Renderer::Vulkan::VulkanRenderer::InitilizeImGUI()
{
	// Setup Window Data
	m_WindowData.Surface = *m_swapchain->GetSurface();
	const VkFormat requestSurfaceImageFormat[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM };
	const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
	m_WindowData.SurfaceFormat = m_swapchain->GetSurfaceFormat();
	m_WindowData.PresentMode = m_swapchain->GetSurfacePresentMode();

	m_WindowData.Swapchain = m_swapchain->GetSwapchain();
	m_WindowData.Width = 1080;
	m_WindowData.Width = 720;
	m_WindowData.RenderPass = *m_swapchain->GetRenderPass();

	//m_WindowData.Framebuffer
	//m_WindowData.ClearValue
	m_WindowData.ClearEnable = false;
	//m_WindowData.BackBufferView
	//m_WindowData.BackBufferCount
	//m_WindowData.BackBuffer

	ImGui_ImplVulkanH_CreateWindowDataCommandBuffers(
		*m_physical_device->GetPhysicalDevice(),
		*m_device->GetVulkanDevice(),
		m_physical_device->GetQueueFamilies()->graphics_indices,
		&m_WindowData,
		NULL
	);

	for (int i = 0; i < m_swapchain->GetImageCount(); i++)
	{
		m_WindowData.Framebuffer[i] = m_swapchain->GetSwapchainFrameBuffers()[i];
	}

	/*ImGui_ImplVulkanH_CreateWindowDataSwapChainAndFramebuffer(
		*m_physical_device->GetPhysicalDevice(),
		*m_device->GetVulkanDevice(),
		&m_WindowData,
		NULL,
		1080,
		720
	);*/

	std::vector<VkDescriptorPoolSize> pool_sizes{
		{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
	{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
	{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
	{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
	{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
	{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
	{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
	{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
	{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
	{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
	{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
	};



	VkDescriptorPool imgui_descriptor_pool = VK_NULL_HANDLE;
	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	pool_info.maxSets = 1000 * pool_sizes.size();
	pool_info.poolSizeCount = (uint32_t)pool_sizes.size();
	pool_info.pPoolSizes = pool_sizes.data();

	vkCreateDescriptorPool(*m_device->GetVulkanDevice(), &pool_info, NULL, &imgui_descriptor_pool);



	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = *m_instance->GetInstance();
	init_info.PhysicalDevice = *m_physical_device->GetPhysicalDevice();
	init_info.Device = *m_device->GetVulkanDevice();
	init_info.QueueFamily = m_physical_device->GetQueueFamilies()->graphics_indices;
	init_info.Queue = *m_device->GetGraphicsQueue();
	init_info.PipelineCache = VK_NULL_HANDLE;
	init_info.DescriptorPool = imgui_descriptor_pool;
	init_info.Allocator = NULL;
	init_info.CheckVkResultFn = NULL;
	ImGui_ImplVulkan_Init(&init_info, m_WindowData.RenderPass);


	
	// Setup fonts
	// Use any command queue
	VkCommandPool command_pool = m_WindowData.Frames[m_WindowData.FrameIndex].CommandPool;
	VkCommandBuffer command_buffer = m_WindowData.Frames[m_WindowData.FrameIndex].CommandBuffer;

	vkResetCommandPool(*m_device->GetVulkanDevice(), command_pool, 0);
	VkCommandBufferBeginInfo begin_info = {};
	begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(command_buffer, &begin_info);

	ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

	VkSubmitInfo end_info = {};
	end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	end_info.commandBufferCount = 1;
	end_info.pCommandBuffers = &command_buffer;
	vkEndCommandBuffer(command_buffer);

	vkQueueSubmit(*m_device->GetGraphicsQueue(), 1, &end_info, VK_NULL_HANDLE);

	vkDeviceWaitIdle(*m_device->GetVulkanDevice());
	ImGui_ImplVulkan_InvalidateFontUploadObjects();

	
}

void Renderer::Vulkan::VulkanRenderer::RenderImGUI()
{
	ImGui::Render();

	VkSemaphore& image_acquired_semaphore = m_WindowData.Frames[m_WindowData.FrameIndex].ImageAcquiredSemaphore;
	vkAcquireNextImageKHR(*m_device->GetVulkanDevice(), m_WindowData.Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &m_WindowData.FrameIndex);
	VkResult status;
	ImGui_ImplVulkanH_FrameData* fd = &m_WindowData.Frames[m_WindowData.FrameIndex];
	{
		status = vkWaitForFences(*m_device->GetVulkanDevice(), 1, &fd->Fence, VK_TRUE, UINT64_MAX);	// wait indefinitely instead of periodically checking

		status = vkResetFences(*m_device->GetVulkanDevice(), 1, &fd->Fence);
	}
	{
		status = vkResetCommandPool(*m_device->GetVulkanDevice(), fd->CommandPool, 0);
		VkCommandBufferBeginInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		status = vkBeginCommandBuffer(fd->CommandBuffer, &info);
	}
	{
		std::array<VkClearValue, 2> clear_values;
		std::copy(std::begin(m_window_handle->clear_color.float32), std::end(m_window_handle->clear_color.float32), std::begin(clear_values[0].color.float32));
		clear_values[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		info.renderPass = m_WindowData.RenderPass;
		info.framebuffer = m_WindowData.Framebuffer[m_WindowData.FrameIndex];
		info.renderArea.extent.width = m_WindowData.Width;
		info.renderArea.extent.height = m_WindowData.Height;
		info.clearValueCount = clear_values.size();
		info.pClearValues = clear_values.data();
		vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	// Record Imgui Draw Data and draw funcs into command buffer
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), fd->CommandBuffer);

	// Submit command buffer
	vkCmdEndRenderPass(fd->CommandBuffer);
	{
		VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		info.waitSemaphoreCount = 1;
		info.pWaitSemaphores = &image_acquired_semaphore;
		info.pWaitDstStageMask = &wait_stage;
		info.commandBufferCount = 1;
		info.pCommandBuffers = &fd->CommandBuffer;
		info.signalSemaphoreCount = 1;
		info.pSignalSemaphores = &fd->RenderCompleteSemaphore;

		status = vkEndCommandBuffer(fd->CommandBuffer);

		status = vkQueueSubmit(*m_device->GetGraphicsQueue(), 1, &info, fd->Fence);
	}






	VkResult present_result = VkResult::VK_RESULT_MAX_ENUM;
	VkPresentInfoKHR present_info = {};
	present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = &fd->RenderCompleteSemaphore;
	present_info.swapchainCount = 1;
	present_info.pSwapchains = &m_WindowData.Swapchain;
	present_info.pImageIndices = &m_WindowData.FrameIndex;
	present_info.pResults = nullptr;

	vkQueuePresentKHR(
		*m_device->GetPresentQueue(),
		&present_info
	);




}

void VulkanRenderer::Update()
{
	m_swapchain->Render();
}

void VulkanRenderer::Stop()
{
	delete m_device;
	delete m_physical_device;
	delete m_instance;
}

void Renderer::Vulkan::VulkanRenderer::Rebuild()
{
	m_swapchain->RebuildSwapchain();
}

IUniformBuffer * Renderer::Vulkan::VulkanRenderer::CreateUniformBuffer(void * dataPtr, unsigned int indexSize, unsigned int elementCount)
{
	return new VulkanUniformBuffer(m_device, dataPtr, indexSize, elementCount);
}

IVertexBuffer * Renderer::Vulkan::VulkanRenderer::CreateVertexBuffer(void * dataPtr, unsigned int indexSize, unsigned int elementCount)
{
	return new VulkanVertexBuffer(m_device, dataPtr, indexSize, elementCount);
}

IIndexBuffer * Renderer::Vulkan::VulkanRenderer::CreateIndexBuffer(void * dataPtr, unsigned int indexSize, unsigned int elementCount)
{
	return new VulkanIndexBuffer(m_device, dataPtr, indexSize, elementCount);
}

IGraphicsPipeline * Renderer::Vulkan::VulkanRenderer::CreateGraphicsPipeline(std::map<ShaderStage, const char*> paths)
{
	VulkanGraphicsPipeline* graphics_pipeline = new VulkanGraphicsPipeline(m_device, m_swapchain, paths);
	m_swapchain->AttachGraphicsPipeline(graphics_pipeline);
	return graphics_pipeline;
}

IComputePipeline * Renderer::Vulkan::VulkanRenderer::CreateComputePipeline(const char * path, unsigned int x, unsigned int y, unsigned int z)
{
	return new VulkanComputePipeline(m_device,path, x, y, z);
}

IComputeProgram * Renderer::Vulkan::VulkanRenderer::CreateComputeProgram()
{
	return new VulkanComputeProgram(m_device);
}

IModelPool * Renderer::Vulkan::VulkanRenderer::CreateModelPool(IVertexBuffer * vertex_buffer, IIndexBuffer * index_buffer)
{
	return new VulkanModelPool(m_device, vertex_buffer, index_buffer);
}

ITextureBuffer * Renderer::Vulkan::VulkanRenderer::CreateTextureBuffer(void * dataPtr, DataFormat format, unsigned int width, unsigned int height)
{
	return new VulkanTextureBuffer(m_device, dataPtr, format, width, height);
}

IDescriptor * Renderer::Vulkan::VulkanRenderer::CreateDescriptor(DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding)
{
	return new VulkanDescriptor(descriptor_type, shader_stage, binding);
}

IDescriptorPool * Renderer::Vulkan::VulkanRenderer::CreateDescriptorPool(std::vector<IDescriptor*> descriptors)
{
	return new VulkanDescriptorPool(m_device, descriptors);
}

VkDescriptorType Renderer::Vulkan::VulkanRenderer::ToDescriptorType(DescriptorType descriptor_type)
{
	switch (descriptor_type)
	{
	case UNIFORM:
		return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	case IMAGE_SAMPLER:
		return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	}
	return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

VkShaderStageFlagBits Renderer::Vulkan::VulkanRenderer::ToVulkanShader(ShaderStage stage)
{
	switch(stage)
	{
		case VERTEX_SHADER:
			return VK_SHADER_STAGE_VERTEX_BIT;
		case FRAGMENT_SHADER:
			return VK_SHADER_STAGE_FRAGMENT_BIT;
		case COMPUTE_SHADER:
			return VK_SHADER_STAGE_COMPUTE_BIT;
		case GEOMETRY_SHADER:
			return VK_SHADER_STAGE_GEOMETRY_BIT;
	}
	return VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM;
}

void Renderer::Vulkan::VulkanRenderer::CreateSurface(Renderer::NativeWindowHandle* window_handle)
{
	auto CreateWin32SurfaceKHR = (PFN_vkCreateWin32SurfaceKHR)vkGetInstanceProcAddr(*m_instance->GetInstance(), "vkCreateWin32SurfaceKHR");

	VkWin32SurfaceCreateInfoKHR createInfo = VulkanInitializers::SurfaceCreateInfo(window_handle);

	if (!CreateWin32SurfaceKHR || CreateWin32SurfaceKHR(*m_instance->GetInstance(), &createInfo, nullptr, &m_surface) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create window surface!");
	}
}

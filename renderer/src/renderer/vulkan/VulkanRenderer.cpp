#include <renderer/vulkan/VulkanRenderer.hpp>
#include <renderer/vulkan/VulkanInstance.hpp>
#include <renderer\vulkan\VulkanFlags.hpp>
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
#include <renderer\vulkan\VulkanDescriptorSet.hpp>
#include <renderer\vulkan\VulkanCommon.hpp>
#include <renderer\vulkan\VulkanRaytracePipeline.hpp>
#include <renderer\vulkan\VulkanAcceleration.hpp>
#include <renderer\vulkan\VulkanRenderPass.hpp>

#include <renderer/NativeWindowHandle.hpp>

#include <assert.h>

using namespace Renderer;
using namespace Renderer::Vulkan;
VulkanRenderer::VulkanRenderer()
{
}

VulkanRenderer::~VulkanRenderer()
{
	Stop();
}

bool Renderer::Vulkan::VulkanRenderer::Start(Renderer::NativeWindowHandle* window_handle, unsigned int flags)
{
	m_window_handle = window_handle;

	m_instance = new VulkanInstance(flags);
	Status::ErrorCheck(m_instance);
	if (HasError())return false;

	CreateSurface(window_handle);

	m_physical_device = VulkanPhysicalDevice::GetPhysicalDevice(m_instance, m_surface);
	Status::ErrorCheck(m_physical_device);
	if (HasError())return false;

	m_device = new VulkanDevice(m_instance, m_physical_device);
	Status::ErrorCheck(m_device);
	if (HasError())return false;

	m_swapchain = new VulkanSwapchain(this, m_instance, m_device, &m_surface, window_handle);
	Status::ErrorCheck(m_swapchain);
	if (HasError())return false;

	m_running = true;
	return m_running;
}

bool VulkanRenderer::Start(Renderer::NativeWindowHandle* window_handle)
{
	return Start(window_handle, Renderer::Vulkan::VulkanFlags::None);
}

void VulkanRenderer::Stop()
{
	if (!m_running)return;
	delete m_swapchain;
	delete m_device;
	delete m_physical_device;
	delete m_instance;
	m_running = false;
}

void Renderer::Vulkan::VulkanRenderer::Rebuild()
{
	m_swapchain->RebuildSwapchain();
}

VulkanUniformBuffer * Renderer::Vulkan::VulkanRenderer::CreateUniformBuffer(void * dataPtr, BufferChain level, unsigned int indexSize, unsigned int elementCount, bool modifiable)
{
	return new VulkanUniformBuffer(m_device, level, dataPtr, indexSize, elementCount, modifiable);
}

VulkanVertexBuffer * Renderer::Vulkan::VulkanRenderer::CreateVertexBuffer(void * dataPtr, unsigned int indexSize, unsigned int elementCount)
{
	return new VulkanVertexBuffer(m_device,  dataPtr, indexSize, elementCount);
}

VulkanIndexBuffer * Renderer::Vulkan::VulkanRenderer::CreateIndexBuffer(void * dataPtr, unsigned int indexSize, unsigned int elementCount)
{
	return new VulkanIndexBuffer(m_device, dataPtr, indexSize, elementCount);
}

VulkanGraphicsPipeline * Renderer::Vulkan::VulkanRenderer::CreateGraphicsPipeline(VulkanRenderPass* render_pass, std::vector<std::pair<VkShaderStageFlagBits, const char*>> paths)
{
	VulkanGraphicsPipeline* graphics_pipeline = new VulkanGraphicsPipeline(m_device, render_pass, paths);
	return graphics_pipeline;
}

VulkanComputePipeline * Renderer::Vulkan::VulkanRenderer::CreateComputePipeline(const char * path, unsigned int x, unsigned int y, unsigned int z)
{
	return new VulkanComputePipeline(m_device,path, x, y, z);
}

VulkanComputeProgram * Renderer::Vulkan::VulkanRenderer::CreateComputeProgram()
{
	return new VulkanComputeProgram(m_device);
}

VulkanModelPool * Renderer::Vulkan::VulkanRenderer::CreateModelPool(VulkanVertexBuffer * vertex_buffer, unsigned int vertex_offset, unsigned int vertex_size, VulkanIndexBuffer * index_buffer, unsigned int index_offset, unsigned int index_size, ModelPoolUsage usage)
{
	return new VulkanModelPool(m_device, vertex_buffer, vertex_offset, vertex_size, index_buffer, index_offset, index_size, usage);
}

VulkanModelPool * Renderer::Vulkan::VulkanRenderer::CreateModelPool(VulkanVertexBuffer * vertex_buffer, unsigned int vertex_offset, unsigned int vertex_size, ModelPoolUsage usage = SingleMesh)
{
	return new VulkanModelPool(m_device, vertex_buffer, vertex_offset, vertex_size, usage);
}

VulkanTextureBuffer * Renderer::Vulkan::VulkanRenderer::CreateTextureBuffer(void * dataPtr, VkFormat format, unsigned int width, unsigned int height, VkImageUsageFlags usageFlags)
{
	return new VulkanTextureBuffer(m_device, BufferChain::Single, dataPtr, format, width, height, usageFlags);
}

VulkanTextureBuffer * Renderer::Vulkan::VulkanRenderer::CreateTextureBuffer(void * dataPtr, BufferChain level, VkFormat format, unsigned int width, unsigned int height, VkImageUsageFlags usageFlags)
{
	return new VulkanTextureBuffer(m_device, level,  dataPtr, format, width, height, usageFlags);
}

VulkanTextureBuffer * Renderer::Vulkan::VulkanRenderer::CreateTextureBuffer(VkFormat format, unsigned int width, unsigned int height, VkImageUsageFlags usageFlags)
{
	return new VulkanTextureBuffer(m_device, BufferChain::Single, format, width, height, usageFlags);
}

VulkanTextureBuffer * Renderer::Vulkan::VulkanRenderer::CreateTextureBuffer(BufferChain level, VkFormat format, unsigned int width, unsigned int height, VkImageUsageFlags usageFlags)
{
	return new VulkanTextureBuffer(m_device, level, format, width, height, usageFlags);
}

/*VulkanDescriptor * Renderer::Vulkan::VulkanRenderer::CreateDescriptor(DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding, unsigned int count)
{
	return new VulkanDescriptor(descriptor_type, shader_stage, binding, count);
}*/

VulkanDescriptor * Renderer::Vulkan::VulkanRenderer::CreateDescriptor(VkDescriptorType descriptor_type, VkShaderStageFlags shader_stage, unsigned int binding, unsigned int count)
{
	return new VulkanDescriptor(descriptor_type, shader_stage, binding, count);
}

VulkanDescriptorPool * Renderer::Vulkan::VulkanRenderer::CreateDescriptorPool(std::vector<VulkanDescriptor*> descriptors)
{
	return new VulkanDescriptorPool(m_device, descriptors);
}

VulkanRaytracePipeline * Renderer::Vulkan::VulkanRenderer::CreateRaytracePipeline(VulkanRenderPass* render_pass, std::vector<std::pair<VkShaderStageFlagBits, const char*>> paths, std::vector<std::map<VkShaderStageFlagBits, const char*>> hitgroups)
{
	VulkanRaytracePipeline* graphics_pipeline = new VulkanRaytracePipeline(m_device, m_swapchain, render_pass, paths, hitgroups);
	return graphics_pipeline;
}

VulkanRenderPass * Renderer::Vulkan::VulkanRenderer::CreateRenderPass(unsigned int subpass_count)
{
	return new VulkanRenderPass(this, m_swapchain, m_instance, m_device, subpass_count);
}

VulkanAcceleration * Renderer::Vulkan::VulkanRenderer::CreateAcceleration()
{
	return new VulkanAcceleration(m_device);
}

VulkanSwapchain * Renderer::Vulkan::VulkanRenderer::GetSwapchain()
{
	return m_swapchain;
}

bool Renderer::Vulkan::VulkanRenderer::IsRunning()
{
	return m_running;
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

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

	m_swapchain = new VulkanSwapchain(m_instance, m_device, &m_surface, window_handle);
	Status::ErrorCheck(m_swapchain);
	if (HasError())return false;

	m_running = true;
	return m_running;
}

bool VulkanRenderer::Start(Renderer::NativeWindowHandle* window_handle)
{
	return Start(window_handle, Renderer::Vulkan::VulkanFlags::None);
}


void VulkanRenderer::Update()
{
	BeginFrame();

	EndFrame();
	
}

void Renderer::Vulkan::VulkanRenderer::BeginFrame()
{
	if (!m_running)return;

	if((m_instance->GetFlags()& VulkanFlags::ActiveCMDRebuild) == VulkanFlags::ActiveCMDRebuild)
		m_swapchain->RequestRebuildCommandBuffers();

	m_swapchain->FindNextImageIndex();
}

void Renderer::Vulkan::VulkanRenderer::EndFrame()
{
	unsigned int currentBuffer = m_swapchain->GetCurrentFrameIndex();

	m_swapchain->SubmitQueue(currentBuffer);

	std::vector<VkSemaphore> signal_semaphores{
		m_swapchain->GetRenderFinishedSemaphore()
	};

	m_swapchain->Present(signal_semaphores);
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

VulkanGraphicsPipeline * Renderer::Vulkan::VulkanRenderer::CreateGraphicsPipeline(std::vector<std::pair<VkShaderStageFlagBits, const char*>> paths, bool priority)
{
	VulkanGraphicsPipeline* graphics_pipeline = new VulkanGraphicsPipeline(m_device, m_swapchain, paths);
	m_swapchain->AttachGraphicsPipeline(graphics_pipeline, priority);
	return graphics_pipeline;
}

void Renderer::Vulkan::VulkanRenderer::RemoveGraphicsPipeline(VulkanGraphicsPipeline * pipeline)
{
	m_swapchain->RemoveGraphicsPipeline(pipeline);
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

VulkanTextureBuffer * Renderer::Vulkan::VulkanRenderer::CreateTextureBuffer(void * dataPtr, VkFormat format, unsigned int width, unsigned int height)
{
	return new VulkanTextureBuffer(m_device, dataPtr, format, width, height);
}

VulkanTextureBuffer * Renderer::Vulkan::VulkanRenderer::CreateTextureBuffer(void * dataPtr, BufferChain level, VkFormat format, unsigned int width, unsigned int height)
{
	return new VulkanTextureBuffer(m_device, level,  dataPtr, format, width, height);
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

VulkanRaytracePipeline * Renderer::Vulkan::VulkanRenderer::CreateRaytracePipeline(std::vector<std::pair<VkShaderStageFlagBits, const char*>> paths, std::vector<std::map<VkShaderStageFlagBits, const char*>> hitgroups, bool priority)
{
	VulkanRaytracePipeline* graphics_pipeline = new VulkanRaytracePipeline(m_device, m_swapchain, paths, hitgroups);
	m_swapchain->AttachGraphicsPipeline(graphics_pipeline, priority);
	return graphics_pipeline;
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

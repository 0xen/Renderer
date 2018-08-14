#include <renderer/vulkan/VulkanPipeline.hpp>
#include <renderer/vulkan/VulkanUniformBuffer.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>

Renderer::Vulkan::VulkanPipeline::VulkanPipeline(VulkanDevice * device, const char * path) :
	IPipeline(path)
{
	m_device = device;
}

Renderer::Vulkan::VulkanPipeline::~VulkanPipeline()
{
}

void Renderer::Vulkan::VulkanPipeline::AttachBuffer(IUniformBuffer * buffer)
{
	m_buffers.push_back(dynamic_cast<VulkanUniformBuffer*>(buffer));
}

bool Renderer::Vulkan::VulkanPipeline::Build()
{
	return false;
}

bool Renderer::Vulkan::VulkanPipeline::CreatePipeline()
{
	return false;
}

void Renderer::Vulkan::VulkanPipeline::DestroyPipeline()
{
}

void Renderer::Vulkan::VulkanPipeline::AttachToCommandBuffer(VkCommandBuffer & command_buffer)
{
	
}

bool Renderer::Vulkan::VulkanPipeline::Rebuild()
{
	if (m_pipeline != VK_NULL_HANDLE)DestroyPipeline();
	return CreatePipeline();
}

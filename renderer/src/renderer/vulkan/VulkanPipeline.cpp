#include <renderer/vulkan/VulkanPipeline.hpp>
#include <renderer/vulkan/VulkanUniformBuffer.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>

Renderer::Vulkan::VulkanPipeline::VulkanPipeline(VulkanDevice * device, const char * path) :
	IPipeline(path)
{
	m_device = device;
}

void Renderer::Vulkan::VulkanPipeline::AttachBuffer(IUniformBuffer * buffer)
{

}

void Renderer::Vulkan::VulkanPipeline::Build()
{

}

void Renderer::Vulkan::VulkanPipeline::CreatePipeline()
{
}

void Renderer::Vulkan::VulkanPipeline::DestroyPipeline()
{
}

void Renderer::Vulkan::VulkanPipeline::AttachToCommandBuffer(VkCommandBuffer & command_buffer)
{
	
}

void Renderer::Vulkan::VulkanPipeline::Rebuild()
{
	if (m_pipeline != VK_NULL_HANDLE)DestroyPipeline();
	CreatePipeline();
}

#include <renderer/vulkan/VulkanPipeline.hpp>
#include <renderer/vulkan/VulkanUniformBuffer.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanDescriptorPool.hpp>
#include <renderer/vulkan/VulkanDescriptorSet.hpp>



Renderer::Vulkan::VulkanPipeline::VulkanPipeline(VulkanDevice * device, std::map<ShaderStage, const char*> paths) :
	IPipeline(paths)
{
	m_device = device;
}

Renderer::Vulkan::VulkanPipeline::~VulkanPipeline()
{
}

void Renderer::Vulkan::VulkanPipeline::AttachDescriptorPool(IDescriptorPool * buffer)
{
	m_descriptor_pools.push_back(static_cast<VulkanDescriptorPool*>(buffer));
}

void Renderer::Vulkan::VulkanPipeline::AttachDescriptorSet(IDescriptorSet * descriptor_set)
{
	m_descriptor_sets.push_back(static_cast<VulkanDescriptorSet*>(descriptor_set));
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

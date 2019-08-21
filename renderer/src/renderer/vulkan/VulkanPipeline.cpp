#include <renderer/vulkan/VulkanPipeline.hpp>
#include <renderer/vulkan/VulkanUniformBuffer.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanDescriptorPool.hpp>
#include <renderer/vulkan/VulkanDescriptorSet.hpp>



Renderer::Vulkan::VulkanPipeline::VulkanPipeline(VulkanDevice * device, std::vector<std::pair<VkShaderStageFlagBits, const char*>> paths) : m_paths(paths)
{
	m_device = device;
}

Renderer::Vulkan::VulkanPipeline::~VulkanPipeline()
{
}

void Renderer::Vulkan::VulkanPipeline::AttachDescriptorPool(unsigned int setID, VulkanDescriptorPool * buffer)
{
	m_descriptor_pools[setID] = buffer;
}

void Renderer::Vulkan::VulkanPipeline::AttachDescriptorSet(unsigned int setID, VulkanDescriptorSet* descriptor_set)
{
	m_descriptor_sets[setID] = descriptor_set;
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

VkPipelineLayout & Renderer::Vulkan::VulkanPipeline::GetPipelineLayout()
{
	return m_pipeline_layout;
}

std::vector<std::pair<VkShaderStageFlagBits, const char*>> Renderer::Vulkan::VulkanPipeline::GetPaths()
{
	return m_paths;
}

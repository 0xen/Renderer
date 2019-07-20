#include <renderer/vulkan/VulkanDescriptor.hpp>
#include <renderer/vulkan/VulkanRenderer.hpp>

using namespace Renderer;
using namespace Renderer::Vulkan;

Renderer::Vulkan::VulkanDescriptor::VulkanDescriptor(VkDescriptorType descriptor_type, VkShaderStageFlags shader_stage, unsigned int binding, unsigned int count) :
	m_binding(binding), m_count(count)
{
	m_descriptor_type = descriptor_type;
	m_shader_stage = shader_stage;
}

VkDescriptorType Renderer::Vulkan::VulkanDescriptor::GetDescriptorType()
{
	return m_descriptor_type;
}

VkShaderStageFlags Renderer::Vulkan::VulkanDescriptor::GetShaderStage()
{
	return m_shader_stage;
}

unsigned int Renderer::Vulkan::VulkanDescriptor::GetBinding()
{
	return m_binding;
}

unsigned int Renderer::Vulkan::VulkanDescriptor::GetCount()
{
	return m_count;
}


#include <renderer/vulkan/VulkanDescriptor.hpp>
#include <renderer/vulkan/VulkanRenderer.hpp>

using namespace Renderer;
using namespace Renderer::Vulkan;

Renderer::Vulkan::VulkanDescriptor::VulkanDescriptor(DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding, unsigned int count) :
	m_descriptor_type(descriptor_type), m_shader_stage(shader_stage), m_binding(binding), m_count(count)
{
	m_vulkan_descriptor_type = VulkanRenderer::ToDescriptorType(descriptor_type);
	m_vulkan_shader_stage = VulkanRenderer::ToVulkanShader(shader_stage);
}

Renderer::Vulkan::VulkanDescriptor::VulkanDescriptor(VkDescriptorType descriptor_type, VkShaderStageFlags shader_stage, unsigned int binding, unsigned int count) :
	m_binding(binding), m_count(count)
{
	m_vulkan_descriptor_type = descriptor_type;
	m_vulkan_shader_stage = shader_stage;
}

VkDescriptorType Renderer::Vulkan::VulkanDescriptor::GetVulkanDescriptorType()
{
	return m_vulkan_descriptor_type;
}

VkShaderStageFlags Renderer::Vulkan::VulkanDescriptor::GetVulkanShaderStage()
{
	return m_vulkan_shader_stage;
}

ShaderStage Renderer::Vulkan::VulkanDescriptor::GetShaderStage()
{
	return m_shader_stage;
}

DescriptorType Renderer::Vulkan::VulkanDescriptor::GetDescriptorType()
{
	return m_descriptor_type;
}

unsigned int Renderer::Vulkan::VulkanDescriptor::GetBinding()
{
	return m_binding;
}

unsigned int Renderer::Vulkan::VulkanDescriptor::GetCount()
{
	return m_count;
}


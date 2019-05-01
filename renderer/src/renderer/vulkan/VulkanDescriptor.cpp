#include <renderer/vulkan/VulkanDescriptor.hpp>
#include <renderer/vulkan/VulkanRenderer.hpp>

using namespace Renderer;
using namespace Renderer::Vulkan;

Renderer::Vulkan::VulkanDescriptor::VulkanDescriptor(DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding) :
	IDescriptor(descriptor_type, shader_stage, binding)
{
	m_vulkan_descriptor_type = VulkanRenderer::ToDescriptorType(descriptor_type);
	m_vulkan_shader_stage = VulkanRenderer::ToVulkanShader(shader_stage);
}

Renderer::Vulkan::VulkanDescriptor::VulkanDescriptor(VkDescriptorType descriptor_type, VkShaderStageFlags shader_stage, unsigned int binding) :
	IDescriptor(binding)
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

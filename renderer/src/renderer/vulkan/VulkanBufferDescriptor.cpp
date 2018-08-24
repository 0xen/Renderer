#include <renderer/vulkan/VulkanBufferDescriptor.hpp>

Renderer::Vulkan::VulkanBufferDescriptor::VulkanBufferDescriptor(DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding) :
	IBufferDescriptor(descriptor_type, shader_stage, binding)
{
	m_descriptor_type = ToVulkanDescriptorType(GetDescriptorType());
	m_shader_stage = ToVulkanShaderStage(GetShaderStage());
}

VkShaderStageFlags Renderer::Vulkan::VulkanBufferDescriptor::GetVulkanShaderStage()
{
	return m_shader_stage;
}

VkDescriptorType Renderer::Vulkan::VulkanBufferDescriptor::GetVulkanDescriptorType()
{
	return m_descriptor_type;
}

VkDescriptorImageInfo & Renderer::Vulkan::VulkanBufferDescriptor::GetDescriptorImageInfo()
{
	return m_image_info;
}

VkDescriptorBufferInfo & Renderer::Vulkan::VulkanBufferDescriptor::GetDescriptorBufferInfo()
{
	return m_buffer_info;
}

VkShaderStageFlags Renderer::Vulkan::VulkanBufferDescriptor::ToVulkanShaderStage(ShaderStage shader_stage)
{
	switch (shader_stage)
	{
	case ShaderStage::VERTEX_SHADER:
		return VK_SHADER_STAGE_VERTEX_BIT;
		break;
	case ShaderStage::FRAGMENT_SHADER:
		return VK_SHADER_STAGE_FRAGMENT_BIT;
		break;
	case ShaderStage::COMPUTE_SHADER:
		return VK_SHADER_STAGE_COMPUTE_BIT;
		break;
	case ShaderStage::GEOMETRY_SHADER:
		return VK_SHADER_STAGE_GEOMETRY_BIT;
		break;
	}
	return VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM;
}

VkDescriptorType Renderer::Vulkan::VulkanBufferDescriptor::ToVulkanDescriptorType(DescriptorType descriptor_type)
{
	switch (descriptor_type)
	{
	case DescriptorType::UNIFORM:
		return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		break;
	case DescriptorType::IMAGE_SAMPLER:
		return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		break;
	}
	return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

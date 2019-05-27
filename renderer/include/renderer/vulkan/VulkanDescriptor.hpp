#pragma once

#include <renderer\vulkan\VulkanHeader.hpp>
#include <renderer/IDescriptor.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanDescriptor : public IDescriptor
		{
		public:
			VulkanDescriptor(DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding, unsigned int count);
			VulkanDescriptor(VkDescriptorType descriptor_type, VkShaderStageFlags shader_stage, unsigned int binding, unsigned int count);

			VkDescriptorType GetVulkanDescriptorType();
			VkShaderStageFlags GetVulkanShaderStage();
		private:
			VkDescriptorType m_vulkan_descriptor_type;
			VkShaderStageFlags m_vulkan_shader_stage;
		};

	}
}
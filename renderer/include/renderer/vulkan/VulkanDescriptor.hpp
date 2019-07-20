#pragma once

#include <renderer\vulkan\VulkanHeader.hpp>
#include <renderer\DescriptorType.hpp>
#include <renderer\ShaderStage.hpp>

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

			ShaderStage GetShaderStage();
			DescriptorType GetDescriptorType();
			unsigned int GetBinding();
			unsigned int GetCount();
		private:
			VkDescriptorType m_vulkan_descriptor_type;
			VkShaderStageFlags m_vulkan_shader_stage;

			DescriptorType m_descriptor_type;
			ShaderStage m_shader_stage;
			unsigned int m_binding;
			unsigned int m_count;
		};

	}
}
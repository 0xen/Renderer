#pragma once

#include <renderer\vulkan\VulkanHeader.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanDescriptor
		{
		public:
			VulkanDescriptor(VkDescriptorType descriptor_type, VkShaderStageFlags shader_stage, unsigned int binding, unsigned int count);

			VkDescriptorType GetDescriptorType();
			VkShaderStageFlags GetShaderStage();
			unsigned int GetBinding();
			unsigned int GetCount();
		private:
			VkDescriptorType m_descriptor_type;
			VkShaderStageFlags m_shader_stage;

			unsigned int m_binding;
			unsigned int m_count;
		};

	}
}
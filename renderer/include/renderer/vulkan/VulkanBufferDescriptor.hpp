#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>
#include <renderer/IBufferDescriptor.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanDevice;
		class VulkanBufferDescriptor : public virtual IBufferDescriptor
		{
		public:
			VulkanBufferDescriptor(DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding);
			VkShaderStageFlags GetVulkanShaderStage();
			VkDescriptorType GetVulkanDescriptorType();
			VkDescriptorImageInfo& GetDescriptorImageInfo();
			VkDescriptorBufferInfo& GetDescriptorBufferInfo();
		protected:
			union
			{
				VkDescriptorImageInfo m_image_info;
				VkDescriptorBufferInfo m_buffer_info;
			};
		private:
			static VkShaderStageFlags ToVulkanShaderStage(ShaderStage shader_stage);
			static VkDescriptorType ToVulkanDescriptorType(DescriptorType descriptor_type);
			VkDescriptorType m_descriptor_type;
			VkShaderStageFlags m_shader_stage;
			

		};
	}
}
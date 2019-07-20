#pragma once

#include <renderer\vulkan\VulkanHeader.hpp>
#include <renderer\vulkan\VulkanStatus.hpp>

#include <vector>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanDevice;
		class VulkanDescriptorSet;
		class VulkanDescriptor;
		class VulkanDescriptorPool : public VulkanStatus
		{
		public:
			VulkanDescriptorPool(VulkanDevice * device, std::vector<VulkanDescriptor*> descriptor);
			~VulkanDescriptorPool();
			VkDescriptorPool GetDescriptorPool();
			VkDescriptorSetLayout GetDescriptorSetLayout();
			std::vector<VulkanDescriptor*> GetDescriptors();
			VulkanDescriptorSet * CreateDescriptorSet();
		private:
			VulkanDevice * m_device;
			std::vector<VkDescriptorSetLayoutBinding> m_layout_bindings;
			std::vector<VkDescriptorPoolSize> m_descriptor_pool_sizes;
			std::vector<VkDescriptorSetLayout> m_descriptor_set_layouts;
			std::vector<VulkanDescriptor*> m_descriptor;
			VkDescriptorPool m_descriptor_pool;
			VkDescriptorSetLayout m_descriptor_set_layout;
		};

	}
}
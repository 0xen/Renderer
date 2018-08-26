#pragma once

#include <renderer\vulkan\VulkanHeader.hpp>
#include <renderer\vulkan\VulkanStatus.hpp>
#include <renderer/IDescriptor.hpp>
#include <renderer/IDescriptorPool.hpp>

#include <vector>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanDevice;
		class VulkanDescriptorPool : public IDescriptorPool, public VulkanStatus
		{
		public:
			VulkanDescriptorPool(VulkanDevice * device, std::vector<IDescriptor*> descriptor);
			VkDescriptorPool GetDescriptorPool();
			VkDescriptorSetLayout GetDescriptorSetLayout();
			std::vector<IDescriptor*> GetDescriptors();
			virtual IDescriptorSet * CreateDescriptorSet();
		private:
			VulkanDevice * m_device;
			std::vector<VkDescriptorSetLayoutBinding> m_layout_bindings;
			std::vector<VkDescriptorPoolSize> m_descriptor_pool_sizes;
			std::vector<VkDescriptorSetLayout> m_descriptor_set_layouts;
			std::vector<IDescriptor*> m_descriptor;
			VkDescriptorPool m_descriptor_pool;
			VkDescriptorSetLayout m_descriptor_set_layout;
		};

	}
}
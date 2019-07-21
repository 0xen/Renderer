#pragma once

#include <renderer\vulkan\VulkanHeader.hpp>

#include <map>
#include <vector>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanDevice;
		class VulkanDescriptorPool;
		class VulkanBuffer;
		class VulkanDescriptorSet
		{
		public:
			VulkanDescriptorSet(VulkanDevice* device, VulkanDescriptorPool * descriptor_pool, VkDescriptorSet set);
			VkDescriptorSet& GetDescriptorSet();
			virtual void UpdateSet();
			virtual void AttachBuffer(unsigned int location, VulkanBuffer* buffer);
			void AttachBuffer(unsigned int location, VkDescriptorBufferInfo& descriptorSet);
			void AttachBuffer(unsigned int location, std::vector<VkDescriptorBufferInfo>& descriptorSet);
			void AttachBuffer(unsigned int location, VkWriteDescriptorSetAccelerationStructureNV& descriptorSet);
			void AttachBuffer(unsigned int location, std::vector<VkWriteDescriptorSetAccelerationStructureNV>& descriptorSet);
			void AttachBuffer(unsigned int location, VkDescriptorImageInfo& descriptorSet);
			void AttachBuffer(unsigned int location, std::vector<VkDescriptorImageInfo>& descriptorSet);
		private:
			VulkanDescriptorPool * m_descriptor_pool;
			VulkanDevice* m_device;
			VkDescriptorSet m_descriptor_set;
			std::vector<VkWriteDescriptorSet> m_write_descriptor_sets;
			std::map<unsigned int, VkWriteDescriptorSet> m_write_descriptor_sets_scratch;
		};

	}
}
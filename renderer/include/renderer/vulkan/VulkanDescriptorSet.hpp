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
			void AttachBuffer(unsigned int location, std::vector<VulkanBuffer*> descriptorSet);
			void AttachBuffer(unsigned int location, std::vector<VkWriteDescriptorSetAccelerationStructureNV> descriptorSet);
			void AttachBuffer(unsigned int location, std::vector<VkDescriptorImageInfo> descriptorSet);
			virtual std::vector<VulkanBuffer*> GetBuffers();
			bool HasBufferAtLocation(unsigned int location);
		private:
			VulkanDescriptorPool * m_descriptor_pool;
			VulkanDevice* m_device;
			VkDescriptorSet m_descriptor_set;
			std::map<unsigned int, VulkanBuffer*> m_bufers;
			std::map<unsigned int, std::vector<VkWriteDescriptorSetAccelerationStructureNV>> m_as_structs;
			std::map<unsigned int, std::vector<VkDescriptorImageInfo>> m_images;
			std::map<unsigned int, std::vector<VkDescriptorBufferInfo>> m_buffers_arrays;
			std::vector<VkWriteDescriptorSet> m_write_descriptor_sets;
		};

	}
}
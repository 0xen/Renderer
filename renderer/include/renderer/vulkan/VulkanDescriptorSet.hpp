#pragma once

#include <renderer\vulkan\VulkanHeader.hpp>
#include <renderer/IDescriptorSet.hpp>

#include <map>
#include <vector>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanDevice;
		class VulkanDescriptorPool;
		class VulkanBuffer;
		class VulkanDescriptorSet : public IDescriptorSet
		{
		public:
			VulkanDescriptorSet(VulkanDevice* device, VulkanDescriptorPool * descriptor_pool, VkDescriptorSet set);
			VkDescriptorSet& GetDescriptorSet();
			virtual void UpdateSet();
			virtual void AttachBuffer(unsigned int location, IBuffer* buffer);
			void AttachBuffer(unsigned int location, std::vector<VkWriteDescriptorSetAccelerationStructureNV> descriptorSet);
			virtual std::vector<IBuffer*> GetBuffers();
			bool HasBufferAtLocation(unsigned int location);
		private:
			VulkanDescriptorPool * m_descriptor_pool;
			VulkanDevice* m_device;
			VkDescriptorSet m_descriptor_set;
			std::map<unsigned int, VulkanBuffer*> m_bufers;
			std::map<unsigned int, std::vector<VkWriteDescriptorSetAccelerationStructureNV>> m_as_structs;
			std::vector<VkWriteDescriptorSet> m_write_descriptor_sets;
		};

	}
}
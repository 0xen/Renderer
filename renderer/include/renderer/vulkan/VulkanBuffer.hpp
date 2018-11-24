#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanBufferData.hpp>
#include <renderer\IBuffer.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanDevice;
		class VulkanBuffer : public virtual IBuffer
		{
		public:
			VulkanBuffer(VulkanDevice* device, BufferChain level, void* dataPtr, unsigned int indexSize, unsigned int elementCount, VkBufferUsageFlags _usage, VkMemoryPropertyFlags _memory_propertys_flag);
			~VulkanBuffer();

			virtual void SetData(BufferSlot slot);
			virtual void SetData(BufferSlot slot, unsigned int count);
			virtual void SetData(BufferSlot slot, unsigned int startIndex, unsigned int count);

			virtual void Resize(BufferSlot slot, void * dataPtr, unsigned int elementCount);

			virtual void Swap(BufferSlot s1, BufferSlot s2);

			VulkanBufferData* GetBufferData(BufferSlot slot);
			VkDescriptorImageInfo& GetDescriptorImageInfo(BufferSlot slot);
			VkDescriptorBufferInfo& GetDescriptorBufferInfo(BufferSlot slot);

		protected:
			void CreateBuffer(BufferSlot slot);
			void DestroyBuffer(BufferSlot slot);
			VulkanDevice * m_device;
			VkBufferUsageFlags m_usage;
			VkMemoryPropertyFlags m_memory_propertys_flag;

			struct GpuBufferAllocation
			{
				bool mapped = false;
				VulkanBufferData buffer;
				union
				{
					VkDescriptorImageInfo image_info;
					VkDescriptorBufferInfo buffer_info;
				};
			};
			GpuBufferAllocation* m_gpu_allocation = nullptr;

		};
	}
}
#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanBufferData.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		enum BufferChain
		{
			Single = 0,
			Double = 1,
			Tripple = 2,
			BUFFER_CHAIN_MAX = 3
		};
		enum BufferSlot
		{
			Primary = 0,
			Secondery = 1,
			Tertiary = 2,
			BUFFER_SLOT_MAX = 3
		};
		class VulkanDevice;
		class VulkanBuffer
		{
		public:
			VulkanBuffer(VulkanDevice* device, BufferChain level, void* dataPtr, unsigned int indexSize, unsigned int elementCount, VkBufferUsageFlags _usage, VkMemoryPropertyFlags _memory_propertys_flag);
			~VulkanBuffer();

			void SetData(BufferSlot slot);
			void SetData(BufferSlot slot, unsigned int count);
			void SetData(BufferSlot slot, unsigned int startIndex, unsigned int count);

			void Resize(BufferSlot slot, void * dataPtr, unsigned int elementCount);

			void Transfer(BufferSlot to, BufferSlot from);

			unsigned int GetIndexSize(BufferSlot slot);
			unsigned int GetElementCount(BufferSlot slot);
			void* GetDataPointer(BufferSlot slot);

			VulkanBufferData* GetBufferData(BufferSlot slot);
			VkDescriptorBufferInfo& GetDescriptorBufferInfo(BufferSlot slot);

		protected:
			void CreateBuffer(BufferSlot slot);
			void DestroyBuffer(BufferSlot slot);
			void Flush(BufferSlot slot);
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

			struct BufferLocalAllocation
			{
				void* dataPtr = nullptr;
				unsigned int bufferSize;
				unsigned int indexSize;
				unsigned int elementCount;
			};
			BufferLocalAllocation* m_local_allocation = nullptr;
			BufferChain m_level;
		};
	}
}
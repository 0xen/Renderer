#pragma once

#include <vector>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanBuffer;
		class VulkanBufferPool
		{
		public:
			VulkanBufferPool(VulkanBuffer* buffer);
			~VulkanBufferPool() {}

			template<typename T>
			T* Get(unsigned int index);

			void* GetRaw(unsigned int index);

			unsigned int Allocate();
			void UnAllocate(unsigned int index);

			VulkanBuffer* GetBuffer();
		public:
			unsigned int m_current_allocation_index;
			VulkanBuffer* m_buffer;
			std::vector<unsigned int> m_free_indexs;
		};
		template<typename T>
		inline T * VulkanBufferPool::Get(unsigned int index)
		{
			return reinterpret_cast<T*>(static_cast<char*>(m_buffer->GetDataPointer(BufferSlot::Primary)) + (index * m_buffer->GetIndexSize(BufferSlot::Primary)));
		}
	}
}
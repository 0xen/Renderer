#pragma once

#include <renderer\IDescriptor.hpp>
#include <renderer\IDescriptorSet.hpp>
#include <renderer\IBuffer.hpp>

namespace Renderer
{
	class IBufferPool
	{
	public:
		IBufferPool(IBuffer* buffer);
		virtual ~IBufferPool() {}

		template<typename T>
		T* Get(unsigned int index);

		void* GetRaw(unsigned int index);

		unsigned int Allocate();
		void UnAllocate(unsigned int);

		IBuffer* GetBuffer();
	public:
		unsigned int m_current_allocation_index;
		IBuffer* m_buffer;

	};
	template<typename T>
	inline T * IBufferPool::Get(unsigned int index)
	{
		return reinterpret_cast<T*>(static_cast<char*>(m_buffer->GetDataPointer(BufferSlot::Primary)) + (index * m_buffer->GetIndexSize(BufferSlot::Primary)));
	}
}
#include <renderer/IBufferPool.hpp>

Renderer::IBufferPool::IBufferPool(IBuffer * buffer) : m_buffer(buffer)
{
	m_current_allocation_index = 0;
}

void * Renderer::IBufferPool::GetRaw(unsigned int index)
{
	return static_cast<char*>(m_buffer->GetDataPointer(BufferSlot::Primary)) + (index * m_buffer->GetIndexSize(BufferSlot::Primary));
}

unsigned int Renderer::IBufferPool::Allocate()
{
	return m_current_allocation_index++;
}

void Renderer::IBufferPool::UnAllocate(unsigned int)
{
}

Renderer::IBuffer * Renderer::IBufferPool::GetBuffer()
{
	return m_buffer;
}

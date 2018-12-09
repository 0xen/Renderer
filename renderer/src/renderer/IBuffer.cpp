#include <renderer/IBuffer.hpp>
#include <cstring>

void Renderer::IBuffer::Transfer(BufferSlot to, BufferSlot from)
{
	memcpy(m_local_allocation + (unsigned int)to, m_local_allocation + (unsigned int)from, sizeof(BufferLocalAllocation));
}

unsigned int Renderer::IBuffer::GetIndexSize(BufferSlot slot)
{
	return m_local_allocation[(unsigned int)slot].indexSize;
}

unsigned int Renderer::IBuffer::GetElementCount(BufferSlot slot)
{
	return m_local_allocation[(unsigned int)slot].elementCount;
}

void * Renderer::IBuffer::GetDataPointer(BufferSlot slot)
{
	return m_local_allocation[(unsigned int)slot].dataPtr;
}

Renderer::IBuffer::IBuffer(BufferChain level)
{
	m_local_allocation = new BufferLocalAllocation[(unsigned int)level + 1];
	m_level = level;
}

Renderer::IBuffer::~IBuffer()
{
	delete[] m_local_allocation;
}
#include <renderer/IBuffer.hpp>

unsigned int Renderer::IBuffer::GetIndexSize()
{
	return m_indexSize;
}

unsigned int Renderer::IBuffer::GetElementCount()
{
	return m_elementCount;
}

void * Renderer::IBuffer::GetDataPointer()
{
	return m_dataPtr;
}

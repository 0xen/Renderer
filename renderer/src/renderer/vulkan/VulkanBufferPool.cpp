#include <renderer/vulkan/VulkanBufferPool.hpp>
#include <renderer\vulkan\VulkanBuffer.hpp>

using namespace Renderer::Vulkan;

Renderer::Vulkan::VulkanBufferPool::VulkanBufferPool(VulkanBuffer * buffer) : m_buffer(buffer)
{
	m_current_allocation_index = 0;
}

void * Renderer::Vulkan::VulkanBufferPool::GetRaw(unsigned int index)
{
	return static_cast<char*>(m_buffer->GetDataPointer(BufferSlot::Primary)) + (index * m_buffer->GetIndexSize(BufferSlot::Primary));
}

unsigned int Renderer::Vulkan::VulkanBufferPool::Allocate()
{
	if (m_free_indexs.size() > 0)
	{
		unsigned int index = m_free_indexs[m_free_indexs.size() - 1];
		m_free_indexs.pop_back();
		return index;
	}
	return m_current_allocation_index++;
}

void Renderer::Vulkan::VulkanBufferPool::UnAllocate(unsigned int index)
{
	m_free_indexs.push_back(index);
}

Renderer::Vulkan::VulkanBuffer * Renderer::Vulkan::VulkanBufferPool::GetBuffer()
{
	return m_buffer;
}

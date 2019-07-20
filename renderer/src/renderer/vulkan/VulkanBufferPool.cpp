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
	return m_current_allocation_index++;
}

void Renderer::Vulkan::VulkanBufferPool::UnAllocate(unsigned int)
{
	/*Need doing*/
}

Renderer::Vulkan::VulkanBuffer * Renderer::Vulkan::VulkanBufferPool::GetBuffer()
{
	return m_buffer;
}

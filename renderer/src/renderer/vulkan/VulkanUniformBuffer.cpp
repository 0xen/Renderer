#include <renderer/vulkan/VulkanUniformBuffer.hpp>
#include <renderer/vulkan/VulkanBuffer.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>

using namespace Renderer::Vulkan;

Renderer::Vulkan::VulkanUniformBuffer::VulkanUniformBuffer(VulkanDevice * device, BufferChain level, void * dataPtr, unsigned int indexSize, unsigned int elementCount, bool modifiable) :
	VulkanBuffer(device, level, dataPtr, indexSize, elementCount,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | (modifiable ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : 0) ,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
{
	VkDeviceSize offset = 0;
	for (unsigned int slot = 0; slot <= (unsigned int)level; slot++)
	{
		m_gpu_allocation[slot].buffer_info = VulkanInitializers::DescriptorBufferInfo(m_gpu_allocation[slot].buffer.buffer, (uint32_t)m_gpu_allocation[slot].buffer.size, offset);
	}
}

Renderer::Vulkan::VulkanUniformBuffer::~VulkanUniformBuffer()
{
}

void Renderer::Vulkan::VulkanUniformBuffer::GetData(BufferSlot slot)
{
	memcpy(
		m_local_allocation[slot].dataPtr, 
		m_gpu_allocation[slot].buffer.mapped_memory, 
		(::size_t)m_local_allocation[slot].bufferSize
	);
}

void Renderer::Vulkan::VulkanUniformBuffer::GetData(BufferSlot slot,unsigned int count)
{
	memcpy(
		m_local_allocation[slot].dataPtr,
		m_gpu_allocation[slot].buffer.mapped_memory, 
		(::size_t)m_local_allocation[slot].indexSize * count
	);
}

void Renderer::Vulkan::VulkanUniformBuffer::GetData(BufferSlot slot,unsigned int startIndex, unsigned int count)
{
	memcpy(
		((char*)m_local_allocation[slot].dataPtr) + (startIndex * m_local_allocation[slot].indexSize),
		((char*)m_gpu_allocation[slot].buffer.mapped_memory) + (startIndex * m_local_allocation[slot].indexSize),
		(::size_t)m_local_allocation[slot].indexSize * count
	);
}

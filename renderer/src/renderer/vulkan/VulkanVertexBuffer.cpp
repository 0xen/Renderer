#include <renderer/vulkan/VulkanVertexBuffer.hpp>
#include <renderer\vulkan\VulkanCommon.hpp>


const Renderer::BufferChain Renderer::Vulkan::VulkanVertexBuffer::m_level = BufferChain::Single;

Renderer::Vulkan::VulkanVertexBuffer::VulkanVertexBuffer(VulkanDevice * device, void * dataPtr, unsigned int indexSize, unsigned int elementCount) :
	VulkanBuffer(device, m_level, dataPtr, indexSize, elementCount,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT),
	IVertexBuffer(m_level),
	IBuffer(m_level)
{
	VkDeviceSize offset = 0;

	m_gpu_allocation[m_level].buffer_info =
		VulkanInitializers::DescriptorBufferInfo(m_gpu_allocation[m_level].buffer.buffer, (uint32_t)m_gpu_allocation[m_level].buffer.size, offset);
}

Renderer::Vulkan::VulkanVertexBuffer::~VulkanVertexBuffer()
{
}

void Renderer::Vulkan::VulkanVertexBuffer::SetData(BufferSlot slot)
{
	CreateStageingBuffer(slot);
	m_staging_buffer->SetData(slot);
	VulkanCommon::CopyBuffer(m_device, m_staging_buffer->GetBufferData(slot)->buffer, this->GetBufferData(slot)->buffer, m_local_allocation[slot].bufferSize);
	DestroyStagingBuffer(slot);
}

void Renderer::Vulkan::VulkanVertexBuffer::SetData(BufferSlot slot, unsigned int count)
{
	CreateStageingBuffer(slot);
	m_staging_buffer->SetData(slot, count);
	VulkanCommon::CopyBuffer(m_device, m_staging_buffer->GetBufferData(slot)->buffer, this->GetBufferData(slot)->buffer, m_local_allocation[slot].bufferSize);
	DestroyStagingBuffer(slot);
}

void Renderer::Vulkan::VulkanVertexBuffer::SetData(BufferSlot slot, unsigned int startIndex, unsigned int count)
{
	CreateStageingBuffer(slot);
	m_staging_buffer->SetData(slot, startIndex, count);
	VulkanCommon::CopyBuffer(m_device, m_staging_buffer->GetBufferData(slot)->buffer, this->GetBufferData(slot)->buffer, m_local_allocation[slot].bufferSize);
	DestroyStagingBuffer(slot);
}

void Renderer::Vulkan::VulkanVertexBuffer::CreateStageingBuffer(BufferSlot slot)
{
	m_staging_buffer = new VulkanBuffer(m_device, BufferChain::Single, m_local_allocation[slot].dataPtr, m_local_allocation[slot].indexSize, m_local_allocation[slot].elementCount,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void Renderer::Vulkan::VulkanVertexBuffer::DestroyStagingBuffer(BufferSlot slot)
{
	delete m_staging_buffer;
}
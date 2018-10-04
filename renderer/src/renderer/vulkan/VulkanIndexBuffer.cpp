#include <renderer/vulkan/VulkanIndexBuffer.hpp>
#include <renderer\vulkan\VulkanCommon.hpp>

Renderer::Vulkan::VulkanIndexBuffer::VulkanIndexBuffer(VulkanDevice * device, void * dataPtr, unsigned int indexSize, unsigned int elementCount):
	VulkanBuffer(device, dataPtr, indexSize, elementCount,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
{
	VkDeviceSize offset = 0;
	m_buffer_info = VulkanInitializers::DescriptorBufferInfo(m_buffer.buffer, (uint32_t)m_buffer.size, offset);
}

Renderer::Vulkan::VulkanIndexBuffer::~VulkanIndexBuffer()
{

}

void Renderer::Vulkan::VulkanIndexBuffer::SetData()
{
	CreateStageingBuffer();
	m_staging_buffer->SetData();
	VulkanCommon::CopyBuffer(m_device, m_staging_buffer->GetBufferData()->buffer, this->GetBufferData()->buffer, m_bufferSize);
	DestroyStagingBuffer();
}

void Renderer::Vulkan::VulkanIndexBuffer::SetData(unsigned int count)
{
	CreateStageingBuffer();
	m_staging_buffer->SetData(count);
	VulkanCommon::CopyBuffer(m_device, m_staging_buffer->GetBufferData()->buffer, this->GetBufferData()->buffer, m_bufferSize);
	DestroyStagingBuffer();
}

void Renderer::Vulkan::VulkanIndexBuffer::SetData(unsigned int startIndex, unsigned int count)
{
	CreateStageingBuffer();
	m_staging_buffer->SetData(startIndex, count);
	VulkanCommon::CopyBuffer(m_device, m_staging_buffer->GetBufferData()->buffer, this->GetBufferData()->buffer, m_bufferSize);
	DestroyStagingBuffer();
}

void Renderer::Vulkan::VulkanIndexBuffer::CreateStageingBuffer()
{
	m_staging_buffer = new VulkanBuffer(m_device, m_dataPtr, m_indexSize, m_elementCount,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void Renderer::Vulkan::VulkanIndexBuffer::DestroyStagingBuffer()
{
	delete m_staging_buffer;
}

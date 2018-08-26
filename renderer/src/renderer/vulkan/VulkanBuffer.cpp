#include <renderer/vulkan/VulkanBuffer.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>

Renderer::Vulkan::VulkanBuffer::VulkanBuffer(VulkanDevice * device, void * dataPtr, unsigned int indexSize, unsigned int elementCount, VkBufferUsageFlags usage, VkMemoryPropertyFlags memory_propertys_flag)
{
	m_device = device;
	m_bufferSize = indexSize * elementCount;
	m_dataPtr = dataPtr;
	m_indexSize = indexSize;
	m_elementCount = elementCount;

	VulkanCommon::CreateBuffer(device, m_indexSize * m_elementCount, usage, memory_propertys_flag, m_buffer);
	VulkanCommon::MapBufferMemory(device, m_buffer, m_buffer.size);
}

Renderer::Vulkan::VulkanBuffer::~VulkanBuffer()
{
	VulkanCommon::UnMapBufferMemory(m_device, m_buffer);
	VulkanCommon::DestroyBuffer(m_device, m_buffer);
}

void Renderer::Vulkan::VulkanBuffer::SetData()
{
	memcpy(m_buffer.mapped_memory, m_dataPtr, (::size_t)m_bufferSize);
}

void Renderer::Vulkan::VulkanBuffer::SetData(unsigned int count)
{
	memcpy(m_buffer.mapped_memory, m_dataPtr, (::size_t)m_indexSize * count);
}

void Renderer::Vulkan::VulkanBuffer::SetData(unsigned int startIndex, unsigned int count)
{
	memcpy(((char*)m_buffer.mapped_memory) + (startIndex * m_indexSize), ((char*)m_dataPtr) + (startIndex * m_indexSize), (::size_t)m_indexSize * count);
}

void Renderer::Vulkan::VulkanBuffer::Resize(unsigned int elementCount)
{
}

Renderer::Vulkan::VulkanBufferData * Renderer::Vulkan::VulkanBuffer::GetBufferData()
{
	return &m_buffer;
}

VkDescriptorImageInfo & Renderer::Vulkan::VulkanBuffer::GetDescriptorImageInfo()
{
	return m_image_info;
}

VkDescriptorBufferInfo & Renderer::Vulkan::VulkanBuffer::GetDescriptorBufferInfo()
{
	return m_buffer_info;
}

#include <renderer/vulkan/VulkanBuffer.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>

Renderer::Vulkan::VulkanBuffer::VulkanBuffer(VulkanDevice * device, BufferChain level, void * dataPtr, unsigned int indexSize, unsigned int elementCount, VkBufferUsageFlags usage, VkMemoryPropertyFlags memory_propertys_flag)
{
	m_level = level;
	m_device = device;
	m_gpu_allocation = new GpuBufferAllocation[(unsigned int)level + 1];

	m_usage = usage;
	if (level>BufferChain::Single) m_usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	if (level>BufferChain::Single) m_usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	m_memory_propertys_flag = memory_propertys_flag;

	for (unsigned int slot = 0; slot <= (unsigned int)level; slot++)
	{
		// Setup local GPU data
		m_local_allocation[slot].bufferSize = indexSize * elementCount;
		m_local_allocation[slot].dataPtr = dataPtr;
		m_local_allocation[slot].indexSize = indexSize;
		m_local_allocation[slot].elementCount = elementCount;
		// Setup GPU data
		CreateBuffer((BufferSlot)slot);
		m_gpu_allocation[slot].mapped = true;
	}
}

Renderer::Vulkan::VulkanBuffer::~VulkanBuffer()
{
	for (unsigned int slot = 0; slot <= (unsigned int)m_level; slot++)
	{
		DestroyBuffer((BufferSlot)slot);
	}
	delete[] m_gpu_allocation;
}

void Renderer::Vulkan::VulkanBuffer::SetData(BufferSlot slot)
{
	memcpy(
		m_gpu_allocation[(unsigned int)slot].buffer.mapped_memory, 
		m_local_allocation[(unsigned int)slot].dataPtr,
		(::size_t)m_local_allocation[(unsigned int)slot].bufferSize
	);
	Flush(slot);
}

void Renderer::Vulkan::VulkanBuffer::SetData(BufferSlot slot, unsigned int count)
{
	memcpy(
		m_gpu_allocation[(unsigned int)slot].buffer.mapped_memory, 
		m_local_allocation[(unsigned int)slot].dataPtr, 
		(::size_t)m_local_allocation[(unsigned int)slot].indexSize * count
	);
	Flush(slot);
}

void Renderer::Vulkan::VulkanBuffer::SetData(BufferSlot slot, unsigned int startIndex, unsigned int count)
{
	memcpy(((char*)m_gpu_allocation[(unsigned int)slot].buffer.mapped_memory) + (startIndex * m_local_allocation[(unsigned int)slot].indexSize),
		((char*)m_local_allocation[(unsigned int)slot].dataPtr) + (startIndex * m_local_allocation[(unsigned int)slot].indexSize),
		(::size_t)m_local_allocation[(unsigned int)slot].indexSize * count
	);
	Flush(slot);
}

void Renderer::Vulkan::VulkanBuffer::Resize(BufferSlot slot, void * dataPtr, unsigned int elementCount)
{
	m_local_allocation[(unsigned int)slot].dataPtr = dataPtr;
	m_local_allocation[(unsigned int)slot].bufferSize = m_local_allocation[(unsigned int)slot].indexSize * elementCount;
	m_local_allocation[(unsigned int)slot].elementCount = elementCount;
	DestroyBuffer((BufferSlot)slot);
	CreateBuffer((BufferSlot)slot);
}

void Renderer::Vulkan::VulkanBuffer::Transfer(BufferSlot to, BufferSlot from)
{
	VulkanCommon::CopyBuffer(m_device, m_gpu_allocation[from].buffer.buffer, m_gpu_allocation[to].buffer.buffer, m_local_allocation[to].bufferSize);
}

unsigned int Renderer::Vulkan::VulkanBuffer::GetIndexSize(BufferSlot slot)
{
	return m_local_allocation[(unsigned int)slot].indexSize;
}

unsigned int Renderer::Vulkan::VulkanBuffer::GetElementCount(BufferSlot slot)
{
	return m_local_allocation[(unsigned int)slot].elementCount;
}

void * Renderer::Vulkan::VulkanBuffer::GetDataPointer(BufferSlot slot)
{
	return m_local_allocation[(unsigned int)slot].dataPtr;
}

Renderer::Vulkan::VulkanBufferData * Renderer::Vulkan::VulkanBuffer::GetBufferData(BufferSlot slot)
{
	return &m_gpu_allocation[slot].buffer;
}

VkDescriptorImageInfo & Renderer::Vulkan::VulkanBuffer::GetDescriptorImageInfo(BufferSlot slot)
{
	return m_gpu_allocation[slot].image_info;
}

VkDescriptorBufferInfo & Renderer::Vulkan::VulkanBuffer::GetDescriptorBufferInfo(BufferSlot slot)
{
	return m_gpu_allocation[slot].buffer_info;
}

void Renderer::Vulkan::VulkanBuffer::CreateBuffer(BufferSlot slot)
{
	VulkanCommon::CreateBuffer(
		m_device,
		m_local_allocation[slot].indexSize * m_local_allocation[slot].elementCount,
		m_usage, 
		m_memory_propertys_flag,
		m_gpu_allocation[slot].buffer
	);
	if ((m_memory_propertys_flag & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
	{
		VulkanCommon::MapBufferMemory(m_device, m_gpu_allocation[slot].buffer, m_gpu_allocation[slot].buffer.size);
	}
}

void Renderer::Vulkan::VulkanBuffer::DestroyBuffer(BufferSlot slot)
{
	if (m_gpu_allocation[slot].mapped)
	{
		VulkanCommon::UnMapBufferMemory(m_device, m_gpu_allocation[slot].buffer);
	}
	VulkanCommon::DestroyBuffer(m_device, m_gpu_allocation[slot].buffer);
}

void Renderer::Vulkan::VulkanBuffer::Flush(BufferSlot slot)
{
	VkMappedMemoryRange mappedRange = {};
	mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
	mappedRange.memory = m_gpu_allocation[(unsigned int)slot].buffer.buffer_memory;
	mappedRange.offset = 0;
	mappedRange.size = VK_WHOLE_SIZE;
	VkResult res = vkFlushMappedMemoryRanges(*m_device->GetVulkanDevice(), 1, &mappedRange);
}
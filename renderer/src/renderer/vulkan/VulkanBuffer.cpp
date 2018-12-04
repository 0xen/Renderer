#include <renderer/vulkan/VulkanBuffer.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>

Renderer::Vulkan::VulkanBuffer::VulkanBuffer(VulkanDevice * device, BufferChain level, void * dataPtr, unsigned int indexSize, unsigned int elementCount, VkBufferUsageFlags usage, VkMemoryPropertyFlags memory_propertys_flag) :
	IBuffer(level)
{
	m_device = device;
	m_gpu_allocation = new GpuBufferAllocation[(unsigned int)level + 1];

	m_usage = usage;
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
	VkMappedMemoryRange memoryRange = VulkanInitializers::MappedMemoryRange(m_gpu_allocation[(unsigned int)slot].buffer.buffer_memory, m_gpu_allocation[(unsigned int)slot].buffer.size);
	vkFlushMappedMemoryRanges(*m_device->GetVulkanDevice(), 1, &memoryRange);
}

void Renderer::Vulkan::VulkanBuffer::SetData(BufferSlot slot, unsigned int count)
{
	memcpy(
		m_gpu_allocation[(unsigned int)slot].buffer.mapped_memory, 
		m_local_allocation[(unsigned int)slot].dataPtr, 
		(::size_t)m_local_allocation[(unsigned int)slot].indexSize * count
	);
	VkMappedMemoryRange memoryRange = VulkanInitializers::MappedMemoryRange(m_gpu_allocation[(unsigned int)slot].buffer.buffer_memory, m_gpu_allocation[(unsigned int)slot].buffer.size);
	vkFlushMappedMemoryRanges(*m_device->GetVulkanDevice(), 1, &memoryRange);
}

void Renderer::Vulkan::VulkanBuffer::SetData(BufferSlot slot, unsigned int startIndex, unsigned int count)
{
	memcpy(((char*)m_gpu_allocation[(unsigned int)slot].buffer.mapped_memory) + (startIndex * m_local_allocation[(unsigned int)slot].indexSize),
		((char*)m_local_allocation[(unsigned int)slot].dataPtr) + (startIndex * m_local_allocation[(unsigned int)slot].indexSize),
		(::size_t)m_local_allocation[(unsigned int)slot].indexSize * count
	);
	VkMappedMemoryRange memoryRange = VulkanInitializers::MappedMemoryRange(m_gpu_allocation[(unsigned int)slot].buffer.buffer_memory, m_gpu_allocation[(unsigned int)slot].buffer.size);
	vkFlushMappedMemoryRanges(*m_device->GetVulkanDevice(), 1, &memoryRange);
}

void Renderer::Vulkan::VulkanBuffer::Resize(BufferSlot slot, void * dataPtr, unsigned int elementCount)
{
	m_local_allocation[(unsigned int)slot].dataPtr = dataPtr;
	m_local_allocation[(unsigned int)slot].bufferSize = m_local_allocation[(unsigned int)slot].indexSize * elementCount;
	m_local_allocation[(unsigned int)slot].elementCount = elementCount;
	DestroyBuffer((BufferSlot)slot);
	CreateBuffer((BufferSlot)slot);
}

void Renderer::Vulkan::VulkanBuffer::Swap(BufferSlot s1, BufferSlot s2)
{
	IBuffer::Swap(s1, s2);
	memcpy(m_gpu_allocation + (unsigned int)s1, m_gpu_allocation + (unsigned int)s2, sizeof(GpuBufferAllocation));
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
	VulkanCommon::MapBufferMemory(m_device, m_gpu_allocation[slot].buffer, m_gpu_allocation[slot].buffer.size);
}

void Renderer::Vulkan::VulkanBuffer::DestroyBuffer(BufferSlot slot)
{
	if (m_gpu_allocation[slot].mapped)
	{
		VulkanCommon::UnMapBufferMemory(m_device, m_gpu_allocation[slot].buffer);
	}
	VulkanCommon::DestroyBuffer(m_device, m_gpu_allocation[slot].buffer);
}

#include <renderer/vulkan/VulkanUniformBuffer.hpp>
#include <renderer/vulkan/VulkanBuffer.hpp>
#include <renderer/vulkan/VulkanBufferDescriptor.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>

using namespace Renderer::Vulkan;

Renderer::Vulkan::VulkanUniformBuffer::VulkanUniformBuffer(VulkanDevice * device, void * dataPtr, unsigned int indexSize, unsigned int elementCount, DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding) :
	VulkanBuffer(device, dataPtr, indexSize, elementCount,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
	VulkanBufferDescriptor(descriptor_type, shader_stage, binding), 
	IBufferDescriptor(descriptor_type, shader_stage, binding)
{
	VkDeviceSize offset = 0;
	m_buffer_info = VulkanInitializers::DescriptorBufferInfo(m_buffer.buffer, (uint32_t)m_buffer.size, offset);
}

Renderer::Vulkan::VulkanUniformBuffer::~VulkanUniformBuffer()
{
}

void Renderer::Vulkan::VulkanUniformBuffer::GetData()
{
	memcpy(m_dataPtr, m_buffer.mapped_memory, (::size_t)m_bufferSize);
}

void Renderer::Vulkan::VulkanUniformBuffer::GetData(unsigned int count)
{
	memcpy(m_dataPtr, m_buffer.mapped_memory, (::size_t)m_indexSize * count);
}

void Renderer::Vulkan::VulkanUniformBuffer::GetData(unsigned int startIndex, unsigned int count)
{
	memcpy(((char*)m_dataPtr) + (startIndex * m_indexSize), ((char*)m_buffer.mapped_memory) + (startIndex * m_indexSize), (::size_t)m_indexSize * count);
}
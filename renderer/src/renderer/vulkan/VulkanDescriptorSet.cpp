#include <renderer/vulkan/VulkanDescriptorSet.hpp>
#include <renderer/vulkan/VulkanBuffer.hpp>
#include <renderer/vulkan/VulkanDescriptorPool.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanDescriptor.hpp>

using namespace Renderer;
using namespace Renderer::Vulkan;

Renderer::Vulkan::VulkanDescriptorSet::VulkanDescriptorSet(VulkanDevice* device, VulkanDescriptorPool * descriptor_pool, VkDescriptorSet set)
{
	m_device = device;
	m_descriptor_pool = descriptor_pool;
	m_descriptor_set = set;
}

VkDescriptorSet & Renderer::Vulkan::VulkanDescriptorSet::GetDescriptorSet()
{
	return m_descriptor_set;
}

void Renderer::Vulkan::VulkanDescriptorSet::UpdateSet()
{
	VkDeviceSize offset = 0;
	m_write_descriptor_sets.clear();
	for (IDescriptor* descriptor : m_descriptor_pool->GetDescriptors())
	{
		VulkanDescriptor* vulkan_descriptor = static_cast<VulkanDescriptor*>(descriptor);
		if (HasBufferAtLocation(descriptor->GetBinding()))
		{
			VulkanBuffer* buffer = m_bufers[vulkan_descriptor->GetBinding()];
			if (vulkan_descriptor->GetDescriptorType() == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
			{
				m_write_descriptor_sets.push_back(VulkanInitializers::WriteDescriptorSet(m_descriptor_set, buffer->GetDescriptorImageInfo(BufferSlot::Primary), vulkan_descriptor->GetVulkanDescriptorType(), vulkan_descriptor->GetBinding()));
			}
			else
			{
				m_write_descriptor_sets.push_back(VulkanInitializers::WriteDescriptorSet(m_descriptor_set, buffer->GetDescriptorBufferInfo(BufferSlot::Primary), vulkan_descriptor->GetVulkanDescriptorType(), vulkan_descriptor->GetBinding()));
			}
		}
	}
	vkUpdateDescriptorSets(*m_device->GetVulkanDevice(), (uint32_t)m_write_descriptor_sets.size(), m_write_descriptor_sets.data(), 0, NULL);
}

void Renderer::Vulkan::VulkanDescriptorSet::AttachBuffer(unsigned int location, IBuffer * buffer)
{
	m_bufers[location] = dynamic_cast<VulkanBuffer*>(buffer);
}

std::vector<IBuffer*> Renderer::Vulkan::VulkanDescriptorSet::GetBuffers()
{
	std::vector<IBuffer*> buffers;
	for (auto& buffer : m_bufers)
	{
		buffers.push_back(buffer.second);
	}
	return buffers;
}

bool Renderer::Vulkan::VulkanDescriptorSet::HasBufferAtLocation(unsigned int location)
{
	return m_bufers.find(location) != m_bufers.end();
}

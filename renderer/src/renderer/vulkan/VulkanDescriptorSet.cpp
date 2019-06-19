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
			if (vulkan_descriptor->GetVulkanDescriptorType() == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
			{
				m_write_descriptor_sets.push_back(VulkanInitializers::WriteDescriptorSet(m_descriptor_set, buffer->GetDescriptorImageInfo(BufferSlot::Primary), vulkan_descriptor->GetVulkanDescriptorType(), vulkan_descriptor->GetBinding()));
			}
			else
			{
				m_write_descriptor_sets.push_back(VulkanInitializers::WriteDescriptorSet(m_descriptor_set, buffer->GetDescriptorBufferInfo(BufferSlot::Primary), vulkan_descriptor->GetVulkanDescriptorType(), vulkan_descriptor->GetBinding()));
			}
		}
		else if (m_as_structs.find(descriptor->GetBinding()) != m_as_structs.end())
		{
			m_write_descriptor_sets.push_back(VulkanInitializers::WriteDescriptorSet(m_descriptor_set, m_as_structs[descriptor->GetBinding()], vulkan_descriptor->GetVulkanDescriptorType(), vulkan_descriptor->GetBinding()));
		}
		else if (m_images.find(descriptor->GetBinding()) != m_images.end())
		{
			m_write_descriptor_sets.push_back(VulkanInitializers::WriteDescriptorSet(m_descriptor_set, m_images[descriptor->GetBinding()], vulkan_descriptor->GetVulkanDescriptorType(), vulkan_descriptor->GetBinding()));
		}
		else if (m_buffers_arrays.find(descriptor->GetBinding()) != m_buffers_arrays.end())
		{
			m_write_descriptor_sets.push_back(VulkanInitializers::WriteDescriptorSet(m_descriptor_set, m_buffers_arrays[descriptor->GetBinding()], vulkan_descriptor->GetVulkanDescriptorType(), vulkan_descriptor->GetBinding()));
		}
	}
	vkUpdateDescriptorSets(*m_device->GetVulkanDevice(), (uint32_t)m_write_descriptor_sets.size(), m_write_descriptor_sets.data(), 0, NULL);
}

void Renderer::Vulkan::VulkanDescriptorSet::AttachBuffer(unsigned int location, IBuffer * buffer)
{
	m_bufers[location] = dynamic_cast<VulkanBuffer*>(buffer);
}

void Renderer::Vulkan::VulkanDescriptorSet::AttachBuffer(unsigned int location, std::vector<IBuffer*> descriptorSet)
{
	std::vector<VkDescriptorBufferInfo> buffers;
	for (auto& buffer : descriptorSet)
	{
		buffers.push_back(dynamic_cast<VulkanBuffer*>(buffer)->GetDescriptorBufferInfo(BufferSlot::Primary));
	}
	m_buffers_arrays[location] = buffers;
}

void Renderer::Vulkan::VulkanDescriptorSet::AttachBuffer(unsigned int location, std::vector<VkWriteDescriptorSetAccelerationStructureNV> descriptorSet)
{
	m_as_structs[location] = descriptorSet;
}

void Renderer::Vulkan::VulkanDescriptorSet::AttachBuffer(unsigned int location, std::vector<VkDescriptorImageInfo> descriptorSet)
{
	m_images[location] = descriptorSet;
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

#include <renderer/vulkan/VulkanDescriptorSet.hpp>
#include <renderer/vulkan/VulkanBuffer.hpp>
#include <renderer/vulkan/VulkanTextureBuffer.hpp>
#include <renderer/vulkan/VulkanDescriptorPool.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanDescriptor.hpp>

#include <assert.h>

using namespace Renderer;
using namespace Renderer::Vulkan;

Renderer::Vulkan::VulkanDescriptorSet::VulkanDescriptorSet(VulkanDevice* device, VulkanDescriptorPool * descriptor_pool, VkDescriptorSet set)
{
	m_device = device;
	m_descriptor_pool = descriptor_pool;
	m_descriptor_set = set;
	 // Set the size of the sets to the size of the pool
	m_write_descriptor_sets.resize(m_descriptor_pool->GetDescriptors().size());
}

VkDescriptorSet & Renderer::Vulkan::VulkanDescriptorSet::GetDescriptorSet()
{
	return m_descriptor_set;
}

void Renderer::Vulkan::VulkanDescriptorSet::UpdateSet()
{
	vkUpdateDescriptorSets(*m_device->GetVulkanDevice(), (uint32_t)m_write_descriptor_sets.size(), m_write_descriptor_sets.data(), 0, NULL);
}

void Renderer::Vulkan::VulkanDescriptorSet::AttachBuffer(unsigned int location, VulkanBuffer * buffer)
{
	VulkanDescriptor* descriptor = m_descriptor_pool->GetDescriptors()[location];
	if (descriptor->GetDescriptorType() == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
	{
		AttachBuffer(location, { static_cast<VulkanTextureBuffer*>(buffer)->GetDescriptorImageInfo(BufferSlot::Primary) }); // Write as a single image buffer
	}
	else
	{
		AttachBuffer(location, buffer->GetDescriptorBufferInfo(BufferSlot::Primary)); // Write as a single buffer
	}
}

void Renderer::Vulkan::VulkanDescriptorSet::AttachBuffer(unsigned int location, VkDescriptorBufferInfo& descriptorSet)
{
	assert(m_write_descriptor_sets.size() > location && "Descriptor Set Location Out Of Bounds");
	VulkanDescriptor* descriptor = m_descriptor_pool->GetDescriptors()[location];
	m_write_descriptor_sets[location] = VulkanInitializers::WriteDescriptorSet(m_descriptor_set, descriptorSet, descriptor->GetDescriptorType(), location);
}

void Renderer::Vulkan::VulkanDescriptorSet::AttachBuffer(unsigned int location, std::vector<VkWriteDescriptorSetAccelerationStructureNV>& descriptorSet)
{
	assert(m_write_descriptor_sets.size() > location && "Descriptor Set Location Out Of Bounds");
	VulkanDescriptor* descriptor = m_descriptor_pool->GetDescriptors()[location];
	m_write_descriptor_sets[location] = VulkanInitializers::WriteDescriptorSet(m_descriptor_set, descriptorSet, descriptor->GetDescriptorType(), location);
}

void Renderer::Vulkan::VulkanDescriptorSet::AttachBuffer(unsigned int location, VkWriteDescriptorSetAccelerationStructureNV & descriptorSet)
{
	assert(m_write_descriptor_sets.size() > location && "Descriptor Set Location Out Of Bounds");
	VulkanDescriptor* descriptor = m_descriptor_pool->GetDescriptors()[location];
	m_write_descriptor_sets[location] = VulkanInitializers::WriteDescriptorSet(m_descriptor_set, descriptorSet, descriptor->GetDescriptorType(), location);
}

void Renderer::Vulkan::VulkanDescriptorSet::AttachBuffer(unsigned int location, VkDescriptorImageInfo& descriptorSet)
{
	assert(m_write_descriptor_sets.size() > location && "Descriptor Set Location Out Of Bounds");
	VulkanDescriptor* descriptor = m_descriptor_pool->GetDescriptors()[location];
	m_write_descriptor_sets[location] = VulkanInitializers::WriteDescriptorSet(m_descriptor_set, descriptorSet, descriptor->GetDescriptorType(), location);
}

void Renderer::Vulkan::VulkanDescriptorSet::AttachBuffer(unsigned int location, std::vector<VkDescriptorImageInfo>& descriptorSet)
{
	assert(m_write_descriptor_sets.size() > location && "Descriptor Set Location Out Of Bounds");
	VulkanDescriptor* descriptor = m_descriptor_pool->GetDescriptors()[location];
	m_write_descriptor_sets[location] = VulkanInitializers::WriteDescriptorSet(m_descriptor_set, descriptorSet, descriptor->GetDescriptorType(), location);
}

#include <renderer/vulkan/VulkanDescriptorPool.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanDescriptor.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanDescriptorSet.hpp>

#include <assert.h>

using namespace Renderer;
using namespace Renderer::Vulkan;


Renderer::Vulkan::VulkanDescriptorPool::VulkanDescriptorPool(VulkanDevice * device, std::vector<IDescriptor*> descriptor)
{
	m_device = device;
	m_descriptor = descriptor;



	m_descriptor_pool_sizes.clear();
	m_layout_bindings.clear();


	for (IDescriptor* descriptor : m_descriptor)
	{
		VulkanDescriptor* vulkan_descriptor = static_cast<VulkanDescriptor*>(descriptor);
		m_descriptor_pool_sizes.push_back(VulkanInitializers::DescriptorPoolSize(vulkan_descriptor->GetVulkanDescriptorType(), descriptor->GetCount()));
		m_layout_bindings.push_back(VulkanInitializers::DescriptorSetLayoutBinding(vulkan_descriptor->GetVulkanDescriptorType(), vulkan_descriptor->GetVulkanShaderStage(), vulkan_descriptor->GetBinding(), descriptor->GetCount()));
	}


	VkDescriptorPoolCreateInfo create_info = VulkanInitializers::DescriptorPoolCreateInfo(m_descriptor_pool_sizes, 10000);

	ErrorCheck(vkCreateDescriptorPool(
		*m_device->GetVulkanDevice(),
		&create_info,
		nullptr,
		&m_descriptor_pool
	));

	if (HasError())assert(0 && "Unable To Create Descriptor Pool");

	VkDescriptorSetLayoutCreateInfo layout_info = VulkanInitializers::DescriptorSetLayoutCreateInfo(m_layout_bindings);

	ErrorCheck(vkCreateDescriptorSetLayout(
		*m_device->GetVulkanDevice(),
		&layout_info,
		nullptr,
		&m_descriptor_set_layout
	));

	if (HasError())assert(0 && "Unable To Create Descriptor Set Layout");
}

Renderer::Vulkan::VulkanDescriptorPool::~VulkanDescriptorPool()
{
	for (int i = 0; i < m_descriptor.size(); i++)
	{
		delete m_descriptor[i];
	}
	vkDestroyDescriptorSetLayout(
		*m_device->GetVulkanDevice(),
		m_descriptor_set_layout,
		nullptr
	);
	vkDestroyDescriptorPool(
		*m_device->GetVulkanDevice(),
		m_descriptor_pool,
		nullptr
	);

}

VkDescriptorPool Renderer::Vulkan::VulkanDescriptorPool::GetDescriptorPool()
{
	return m_descriptor_pool;
}

VkDescriptorSetLayout Renderer::Vulkan::VulkanDescriptorPool::GetDescriptorSetLayout()
{
	return m_descriptor_set_layout;
}

std::vector<IDescriptor*> Renderer::Vulkan::VulkanDescriptorPool::GetDescriptors()
{
	return m_descriptor;
}

IDescriptorSet * Renderer::Vulkan::VulkanDescriptorPool::CreateDescriptorSet()
{
	VkDescriptorSet descriptor_set;
	m_descriptor_set_layouts.clear();
	m_descriptor_set_layouts.push_back(m_descriptor_set_layout);
	VkDescriptorSetAllocateInfo alloc_info = VulkanInitializers::DescriptorSetAllocateInfo(m_descriptor_set_layouts, m_descriptor_pool);
	ErrorCheck(vkAllocateDescriptorSets(
		*m_device->GetVulkanDevice(),
		&alloc_info,
		&descriptor_set
	));
	if (HasError())assert(0 && "Unable To Create Descriptor Set");
	return new VulkanDescriptorSet(m_device, this, descriptor_set);
}

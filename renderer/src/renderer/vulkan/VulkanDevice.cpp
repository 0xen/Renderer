#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanPhysicalDevice.hpp>
#include <renderer/vulkan/VulkanInstance.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>

#include <assert.h>

Renderer::Vulkan::VulkanDevice::VulkanDevice(VulkanInstance * instance, VulkanPhysicalDevice * physical_device)
{
	m_instance = instance;
	m_physical_device = physical_device;

	std::vector<VkDeviceQueueCreateInfo> queue_create_infos;

	std::vector<uint32_t> unique_queue_families;
	unique_queue_families.push_back(m_physical_device->GetQueueFamilies()->graphics_indices);
	//unique_queue_families.push_back(m_physical_device->GetQueueFamilies()->compute_indices);

	static const float priority = 1.0f;
	for (auto queue_family : unique_queue_families)
	{
		queue_create_infos.push_back(VulkanInitializers::DeviceQueueCreate(queue_family, priority));
	}

	VkDeviceCreateInfo create_info{};

	if ((instance->GetFlags() & VulkanFlags::Raytrace) == VulkanFlags::Raytrace)
	{
		create_info = VulkanInitializers::DeviceCreateInfo(
			queue_create_infos,
			*m_physical_device->GetExtenstions(),
			*m_physical_device->GetDeviceFeatures2()
		);
		create_info.pNext = m_physical_device->GetDeviceFeatures2()->pNext;
	}
	else
	{
		create_info = VulkanInitializers::DeviceCreateInfo(
			queue_create_infos,
			*m_physical_device->GetExtenstions(),
			*m_physical_device->GetDeviceFeatures()
		);
	}
	// Create the device
	ErrorCheck(vkCreateDevice(
		*m_physical_device->GetPhysicalDevice(),
		&create_info,
		nullptr,
		&m_device
	));
	assert(!HasError() && "Unable up create vulkan device");

	vkGetDeviceQueue(
		m_device,
		m_physical_device->GetQueueFamilies()->compute_indices,
		0,
		&m_compute_queue
	);
	vkGetDeviceQueue(
		m_device,
		m_physical_device->GetQueueFamilies()->graphics_indices,
		0,
		&m_graphics_queue
	);
	vkGetDeviceQueue(
		m_device,
		m_physical_device->GetQueueFamilies()->graphics_indices,
		0,
		&m_present_queue
	);
	// Setup command pools
	VkCommandPoolCreateInfo compute_pool_info = VulkanInitializers::CommandPoolCreateInfo(m_physical_device->GetQueueFamilies()->compute_indices, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	ErrorCheck(vkCreateCommandPool(
		m_device,
		&compute_pool_info,
		nullptr,
		&m_compute_command_pool
	));

	assert(!HasError() && "Unable to create compute command pool");

	VkCommandPoolCreateInfo graphics_pool_info = VulkanInitializers::CommandPoolCreateInfo(m_physical_device->GetQueueFamilies()->graphics_indices, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	ErrorCheck(vkCreateCommandPool(
		m_device,
		&graphics_pool_info,
		nullptr,
		&m_graphics_command_pool
	));

	assert(!HasError() && "Unable to create graphics command pool");

}

Renderer::Vulkan::VulkanDevice::~VulkanDevice()
{
	vkDestroyCommandPool(
		m_device,
		m_graphics_command_pool,
		nullptr
	);
	m_graphics_command_pool = VK_NULL_HANDLE;
	vkDestroyCommandPool(
		m_device,
		m_compute_command_pool,
		nullptr
	);
	m_compute_command_pool = VK_NULL_HANDLE;
	vkDestroyDevice(
		m_device,
		nullptr
	);
	m_device = VK_NULL_HANDLE;
}

VkDevice * Renderer::Vulkan::VulkanDevice::GetVulkanDevice()
{
	return &m_device;
}

Renderer::Vulkan::VulkanPhysicalDevice * Renderer::Vulkan::VulkanDevice::GetVulkanPhysicalDevice()
{
	return m_physical_device;
}

VkQueue * Renderer::Vulkan::VulkanDevice::GetGraphicsQueue()
{
	return &m_graphics_queue;
}

VkQueue * Renderer::Vulkan::VulkanDevice::GetPresentQueue()
{
	return &m_present_queue;
}

VkQueue * Renderer::Vulkan::VulkanDevice::GetComputeQueue()
{
	return &m_compute_queue;
}

VkCommandPool * Renderer::Vulkan::VulkanDevice::GetGraphicsCommandPool()
{
	return &m_graphics_command_pool;
}

VkCommandPool * Renderer::Vulkan::VulkanDevice::GetComputeCommandPool()
{
	return &m_compute_command_pool;
}

void Renderer::Vulkan::VulkanDevice::GetGraphicsCommand(VkCommandBuffer * buffers, uint32_t count)
{
	VkCommandBufferAllocateInfo command_buffer_allocate_info = VulkanInitializers::CommandBufferAllocateInfo(*GetGraphicsCommandPool(), count);
	ErrorCheck(vkAllocateCommandBuffers(
		*GetVulkanDevice(),
		&command_buffer_allocate_info,
		buffers
	));
}

void Renderer::Vulkan::VulkanDevice::GetGraphicsCommand(VkCommandBuffer * buffers, bool begin)
{
	GetGraphicsCommand(buffers, (uint32_t)1);
	if (begin)
	{
		VkCommandBufferBeginInfo begin_info = VulkanInitializers::CommandBufferBeginInfo(0);
		vkBeginCommandBuffer(*buffers, &begin_info);
	}
}

void Renderer::Vulkan::VulkanDevice::SubmitGraphicsCommand(VkCommandBuffer * buffers, uint32_t count)
{
	VkSubmitInfo submit_info = VulkanInitializers::SubmitInfo(buffers, count);
	ErrorCheck(vkQueueSubmit(m_graphics_queue, 1, &submit_info, VK_NULL_HANDLE));
	ErrorCheck(vkQueueWaitIdle(m_graphics_queue));
}

void Renderer::Vulkan::VulkanDevice::FreeGraphicsCommand(VkCommandBuffer * buffers, uint32_t count)
{
	vkFreeCommandBuffers(
		*GetVulkanDevice(),
		*GetGraphicsCommandPool(),
		count,
		buffers);
}

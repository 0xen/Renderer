#include <renderer/vulkan/VulkanPhysicalDevice.hpp>
#include <renderer/vulkan/VulkanInstance.hpp>

#include <set>
#include <assert.h>

Renderer::Vulkan::VulkanPhysicalDevice::VulkanPhysicalDevice(VkPhysicalDevice device, VulkanQueueFamilyIndices queue_family, unsigned int flags)
{
	m_device = device;
	m_queue_family = queue_family;

	vkGetPhysicalDeviceProperties(
		m_device,
		&m_physical_device_properties
	);


	// Gives us access to Ray tracing specific data such as maxRecursionDepth and shaderHeaderSize
	m_device_raytracing_properties = {};
	m_device_raytracing_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV;
	m_device_raytracing_properties.pNext = nullptr;
	m_device_raytracing_properties.maxRecursionDepth = 0;
	m_device_raytracing_properties.shaderGroupHandleSize = 0;

	if ((flags & VulkanFlags::Raytrace) == VulkanFlags::Raytrace)
	{
		m_physical_device_properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		m_physical_device_properties2.pNext = &m_device_raytracing_properties;
		m_physical_device_properties2.properties = {};

		vkGetPhysicalDeviceProperties2(
			m_device,
			&m_physical_device_properties2
		);

		m_descIndexFeatures = {};
		m_descIndexFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
			
		m_device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		m_device_features2.pNext = &m_descIndexFeatures;

		vkGetPhysicalDeviceFeatures2(
			m_device,
			&m_device_features2
		);
	}

	
	
	// Get the devices physical features
	vkGetPhysicalDeviceFeatures(
		m_device,
		&m_device_features
	);


	
	
	// Get the GPU's memory props
	vkGetPhysicalDeviceMemoryProperties(
		m_device,
		&m_physical_device_mem_properties
	);
	m_device_extensions = GetDeviceExtenstions(flags);
}

VkPhysicalDevice * Renderer::Vulkan::VulkanPhysicalDevice::GetPhysicalDevice()
{
	return &m_device;
}

Renderer::Vulkan::VulkanQueueFamilyIndices * Renderer::Vulkan::VulkanPhysicalDevice::GetQueueFamilies()
{
	return &m_queue_family;
}

VkPhysicalDeviceRayTracingPropertiesNV * Renderer::Vulkan::VulkanPhysicalDevice::GetPhysicalDeviceRayTracingProperties()
{
	return &m_device_raytracing_properties;
}

VkPhysicalDeviceProperties * Renderer::Vulkan::VulkanPhysicalDevice::GetPhysicalDeviceProperties()
{
	return &m_physical_device_properties;
}

VkPhysicalDeviceProperties2 * Renderer::Vulkan::VulkanPhysicalDevice::GetPhysicalDeviceProperties2()
{
	return &m_physical_device_properties2;
}

VkPhysicalDeviceFeatures * Renderer::Vulkan::VulkanPhysicalDevice::GetDeviceFeatures()
{
	return &m_device_features;
}

VkPhysicalDeviceFeatures2 * Renderer::Vulkan::VulkanPhysicalDevice::GetDeviceFeatures2()
{
	return &m_device_features2;
}

VkPhysicalDeviceMemoryProperties * Renderer::Vulkan::VulkanPhysicalDevice::GetPhysicalDeviceMemoryProperties()
{
	return &m_physical_device_mem_properties;
}

std::vector<const char*>* Renderer::Vulkan::VulkanPhysicalDevice::GetExtenstions()
{
	return &m_device_extensions;
}

VkFormatProperties Renderer::Vulkan::VulkanPhysicalDevice::GetFormatProperties(VkFormat format)
{
	VkFormatProperties format_properties;
	vkGetPhysicalDeviceFormatProperties(m_device, format, &format_properties);
	return format_properties;
}

Renderer::Vulkan::VulkanPhysicalDevice * Renderer::Vulkan::VulkanPhysicalDevice::GetPhysicalDevice(VulkanInstance* instance, VkSurfaceKHR surface)
{
	VulkanPhysicalDevice* device_instance = VK_NULL_HANDLE;
	std::vector<VkPhysicalDevice> devices = GetPhysicalDevices(instance);

	VkPhysicalDevice chosen_device = VK_NULL_HANDLE;
	VulkanQueueFamilyIndices chosen_queue_family;
	// Loop through the devices and choose a suitable one
	for (auto& device : devices)
	{
		VulkanQueueFamilyIndices queue_family;
		if (CheckPhysicalDevice(instance, device) && SupportsQueueFamily(instance,device, queue_family, surface))
		{
			VkPhysicalDeviceProperties temp_physical_device_properties;
			vkGetPhysicalDeviceProperties(
				device,
				&temp_physical_device_properties
			);
			// First if the chosen device is null and the only GPU that reports back is a intergrated gpu then use it untill we find a deticated one
			if (chosen_device == VK_NULL_HANDLE || chosen_device != VK_NULL_HANDLE && temp_physical_device_properties.deviceType == VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			{
				chosen_device = device;
				chosen_queue_family = queue_family;
			}
		}
	}
	assert(chosen_device != VK_NULL_HANDLE && "No suitable device");
	device_instance = new VulkanPhysicalDevice(chosen_device, chosen_queue_family, instance->GetFlags());
	return device_instance;
}

std::vector<VkPhysicalDevice> Renderer::Vulkan::VulkanPhysicalDevice::GetPhysicalDevices(VulkanInstance* instance)
{
	uint32_t device_count = 0;
	// Get the current GPU count

	vkEnumeratePhysicalDevices(
		*instance->GetInstance(),
		&device_count,
		nullptr
	);
	// Check there is a GPU available
	assert(device_count > 0 && "No GPU's available");
	// Get the devices
	std::vector<VkPhysicalDevice> devices(device_count);
	// Get the physical devices
	vkEnumeratePhysicalDevices(
		*instance->GetInstance(),
		&device_count,
		devices.data()
	);
	return devices;
}

std::vector<const char*> Renderer::Vulkan::VulkanPhysicalDevice::GetDeviceExtenstions(unsigned int flags)
{
	std::vector<const char*> extentions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};
	if ((flags & VulkanFlags::Raytrace) == VulkanFlags::Raytrace)
	{
		extentions.push_back(VK_NV_RAY_TRACING_EXTENSION_NAME);
		extentions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
	}
	return extentions;
}

bool Renderer::Vulkan::VulkanPhysicalDevice::CheckPhysicalDevice(VulkanInstance* instance, VkPhysicalDevice & device)
{
	bool supportsExtentions = CheckDeviceExtensionSupport(instance, device);
	// For now i will just return the first one thats supported
	return supportsExtentions;
}

bool Renderer::Vulkan::VulkanPhysicalDevice::CheckDeviceExtensionSupport(VulkanInstance* instance, VkPhysicalDevice & device)
{
	// Get extension count
	uint32_t extension_count;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);
	// Get all extensions
	std::vector<VkExtensionProperties> available_extensions(extension_count);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());
	// Get all extension names
	std::vector<const char*> device_extensions = GetDeviceExtenstions(instance->GetFlags());
	// Put them into a set so we can use the erase function
	std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());
	// Loop through extensions and check 
	for (const auto& extension : available_extensions)
	{
		required_extensions.erase(extension.extensionName);
	}
	// If any extensions are left over, it failed
	return required_extensions.empty();
}

bool Renderer::Vulkan::VulkanPhysicalDevice::SupportsQueueFamily(VulkanInstance * instance, VkPhysicalDevice & device, VulkanQueueFamilyIndices & queue_family_indices, VkSurfaceKHR surface)
{
	// Get queue family count
	uint32_t queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
	// Get queue families
	std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());
	// Loop through and choose right family
	uint32_t i = 0;
	for (const auto& queue_family : queue_families)
	{
		if (queue_family.queueCount > 0)
		{
			queue_family_indices.graphics_indices = i;
			queue_family_indices.compute_indices = i;

			VkBool32 present_support = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(
				device,
				i,
				surface,
				&present_support
			);
			if (present_support)
				queue_family_indices.present_indices = i;
		}
		// Graphics return
		if (queue_family_indices.isComplete()) return true;
		i++;
	}
	return false;
}

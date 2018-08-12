#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>

#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanStatus.hpp>
#include <renderer/vulkan/VulkanQueueFamilyIndices.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanInstance;
		class VulkanPhysicalDevice : public VulkanStatus
		{
		public:
			VulkanPhysicalDevice(VkPhysicalDevice device, VulkanQueueFamilyIndices queue_family);

			VkPhysicalDevice* GetPhysicalDevice();
			VulkanQueueFamilyIndices* GetQueueFamilies();
			VkPhysicalDeviceProperties* GetPhysicalDeviceProperties();
			VkPhysicalDeviceFeatures* GetDeviceFeatures();
			VkPhysicalDeviceMemoryProperties* GetPhysicalDeviceMemoryProperties();
			std::vector<const char*>* GetExtenstions();

			static VulkanPhysicalDevice* GetPhysicalDevice(VulkanInstance* instance, VkSurfaceKHR surface);
			static std::vector<VkPhysicalDevice> GetPhysicalDevices(VulkanInstance* instance);
			static std::vector<const char*> GetDeviceExtenstions();
		private:
			static bool CheckPhysicalDevice(VkPhysicalDevice& device);
			static bool CheckDeviceExtensionSupport(VkPhysicalDevice& device);
			static bool SupportsQueueFamily(VulkanInstance* instance, VkPhysicalDevice& device, VulkanQueueFamilyIndices& queue_family, VkSurfaceKHR surface);

			std::vector<const char*> m_device_extensions;
			VkPhysicalDevice m_device;
			VulkanQueueFamilyIndices m_queue_family;
			VkPhysicalDeviceProperties m_physical_device_properties;
			VkPhysicalDeviceFeatures m_device_features;
			VkPhysicalDeviceMemoryProperties m_physical_device_mem_properties;
		};
	}
}
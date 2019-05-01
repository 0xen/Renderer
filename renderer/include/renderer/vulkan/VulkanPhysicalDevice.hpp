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
		enum VulkanFlags;
		class VulkanPhysicalDevice : public VulkanStatus
		{
		public:
			VulkanPhysicalDevice(VkPhysicalDevice device, VulkanQueueFamilyIndices queue_family, VulkanFlags flags);

			VkPhysicalDevice* GetPhysicalDevice();
			VulkanQueueFamilyIndices* GetQueueFamilies();
			VkPhysicalDeviceRayTracingPropertiesNV* GetPhysicalDeviceRayTracingProperties();
			VkPhysicalDeviceProperties* GetPhysicalDeviceProperties();
			VkPhysicalDeviceProperties2* GetPhysicalDeviceProperties2();
			VkPhysicalDeviceFeatures* GetDeviceFeatures();
			VkPhysicalDeviceFeatures2* GetDeviceFeatures2();
			VkPhysicalDeviceMemoryProperties* GetPhysicalDeviceMemoryProperties();
			std::vector<const char*>* GetExtenstions();
			VkFormatProperties GetFormatProperties(VkFormat format);

			static VulkanPhysicalDevice* GetPhysicalDevice(VulkanInstance* instance, VkSurfaceKHR surface);
			static std::vector<VkPhysicalDevice> GetPhysicalDevices(VulkanInstance* instance);
			static std::vector<const char*> GetDeviceExtenstions(VulkanFlags flags);
		private:
			static bool CheckPhysicalDevice(VulkanInstance* instance, VkPhysicalDevice& device);
			static bool CheckDeviceExtensionSupport(VulkanInstance* instance, VkPhysicalDevice& device);
			static bool SupportsQueueFamily(VulkanInstance* instance, VkPhysicalDevice& device, VulkanQueueFamilyIndices& queue_family, VkSurfaceKHR surface);

			std::vector<const char*> m_device_extensions;
			VkPhysicalDevice m_device;
			VulkanQueueFamilyIndices m_queue_family;
			VkPhysicalDeviceRayTracingPropertiesNV m_device_raytracing_properties;
			VkPhysicalDeviceProperties m_physical_device_properties;
			VkPhysicalDeviceProperties2 m_physical_device_properties2;
			VkPhysicalDeviceFeatures m_device_features;
			VkPhysicalDeviceFeatures2 m_device_features2;
			VkPhysicalDeviceMemoryProperties m_physical_device_mem_properties;
		};
	}
}
#pragma once

#include <vector>

#include <renderer/NativeWindowHandle.hpp>
#include <renderer/vulkan/VulkanHeader.hpp>

namespace Renderer
{
	namespace VulkanInitializers
	{
		VkApplicationInfo ApplicationInfo(const char* app_name, uint32_t app_ver, const char* engine_name, uint32_t engine_ver, uint32_t api_version);

		VkInstanceCreateInfo InstanceCreateInfo(VkApplicationInfo& app_info, std::vector<const char*>& instance_extensions, std::vector<const char*>& instance_layers);

		VkWin32SurfaceCreateInfoKHR SurfaceCreateInfo(Renderer::NativeWindowHandle window_handle);

		VkDeviceQueueCreateInfo DeviceQueueCreate(uint32_t queue_family_index, float queue_priority);

		VkDeviceCreateInfo DeviceCreateInfo(std::vector<VkDeviceQueueCreateInfo>& queue_create_infos, const std::vector<const char*>& device_extensions, VkPhysicalDeviceFeatures& device_features);

		VkCommandPoolCreateInfo CommandPoolCreateInfo(uint32_t queue_family_index, VkCommandPoolCreateFlags flags = 0);

		VkCommandBufferAllocateInfo CommandBufferAllocateInfo(VkCommandPool pool, uint32_t command_buffer_count);

		VkCommandBufferBeginInfo CommandBufferBeginInfo(VkCommandBufferUsageFlags flag);

		VkSubmitInfo SubmitInfo(VkCommandBuffer& buffer);

		VkSubmitInfo SubmitInfo(VkCommandBuffer* buffer, uint32_t count = 1);







	}
}
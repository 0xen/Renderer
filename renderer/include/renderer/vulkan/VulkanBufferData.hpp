#pragma once

#include <renderer\vulkan\VulkanBufferData.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		struct VulkanBufferData
		{
			VkBuffer buffer = VK_NULL_HANDLE;
			VkDeviceSize size = VK_NULL_HANDLE;
			VkDeviceSize alignment = VK_NULL_HANDLE;
			VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
			void* mapped_memory = nullptr;
		};
	}
}
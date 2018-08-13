#pragma once

#include <renderer\vulkan\VulkanBufferData.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		struct VulkanBufferData
		{
			VkBuffer buffer;
			VkDeviceSize size;
			VkDeviceSize alignment;
			VkDeviceMemory buffer_memory;
			void* mapped_memory = nullptr;
		};
	}
}
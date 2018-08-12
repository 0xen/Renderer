#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		struct VulkanQueueFamilyIndices
		{
			uint32_t graphics_indices = UINT32_MAX;
			uint32_t present_indices = UINT32_MAX;
			uint32_t compute_indices = UINT32_MAX;
			bool isComplete()
			{
				return graphics_indices < UINT32_MAX &&
					present_indices < UINT32_MAX &&
					compute_indices < UINT32_MAX;
			}
		};
	}
}
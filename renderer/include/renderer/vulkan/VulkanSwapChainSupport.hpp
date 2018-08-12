#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>

#include <vector>

namespace Renderer
{
	namespace Vulkan
	{
		struct VulkanSwapChainSupport
		{
			VkSurfaceCapabilitiesKHR capabilities;
			std::vector<VkPresentModeKHR> present_modes;
			std::vector<VkSurfaceFormatKHR> formats;
		};
	}
}
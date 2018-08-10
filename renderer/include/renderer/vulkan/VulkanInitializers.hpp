#pragma once

#include <vector>

#include <vulkan\vulkan.h>

namespace Renderer
{
	namespace VulkanInitializers
	{
		VkApplicationInfo ApplicationInfo(const char* app_name, uint32_t app_ver, const char* engine_name, uint32_t engine_ver, uint32_t api_version);

		VkInstanceCreateInfo InstanceCreateInfo(VkApplicationInfo& app_info, std::vector<const char*>& instance_extensions, std::vector<const char*>& instance_layers);

	}
}
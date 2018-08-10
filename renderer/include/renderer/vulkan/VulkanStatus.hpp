#pragma once

#include <vulkan\vulkan.h>

#include <renderer\Status.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanStatus : public Status
		{
		public:
			VulkanStatus() : Status() {}
		protected:
			bool ErrorCheck(VkResult res)
			{
				m_status = VkResult::VK_SUCCESS != res;
				return HasError();
			}
		};
	}
}
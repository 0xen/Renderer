#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>

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
				return Status::ErrorCheck(VkResult::VK_SUCCESS != res);
			}
			void vErrorCheck(VkResult res)
			{

			}
		};
	}
}
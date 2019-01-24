#pragma once

#include <renderer\IComputeProgram.hpp>
#include <renderer\vulkan\VulkanStatus.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanDevice;
		class VulkanComputeProgram : public IComputeProgram, public VulkanStatus
		{
		public:
			VulkanComputeProgram(VulkanDevice * device);
			virtual ~VulkanComputeProgram();
			virtual void Build();
			virtual void Run();
		private:
			VulkanDevice * m_device;
			VkFence m_fence;
			VkCommandBuffer m_command_buffer;
		};
	}
}
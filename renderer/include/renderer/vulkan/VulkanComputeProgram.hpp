#pragma once

#include <renderer\vulkan\VulkanStatus.hpp>

#include <vector>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanDevice;
		class VulkanComputePipeline;
		class VulkanComputeProgram : public VulkanStatus
		{
		public:
			VulkanComputeProgram(VulkanDevice * device);
			~VulkanComputeProgram();
			void Build();
			void Run();
			void AttachPipeline(VulkanComputePipeline* pipeline);
		private:
			VulkanDevice * m_device;
			VkFence m_fence;
			VkCommandBuffer m_command_buffer;
			std::vector<VulkanComputePipeline*> m_pipelines;
		};
	}
}
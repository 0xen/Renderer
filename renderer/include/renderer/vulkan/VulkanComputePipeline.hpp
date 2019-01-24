#pragma once

#include <renderer\IComputePipeline.hpp>
#include <renderer\vulkan\VulkanPipeline.hpp>
#include <renderer\vulkan\VulkanStatus.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanComputePipeline : public IComputePipeline, public VulkanPipeline, public VulkanStatus
		{
		public:
			VulkanComputePipeline(VulkanDevice * device, const char* path, unsigned int x, unsigned int y, unsigned int z);
			virtual ~VulkanComputePipeline();
			virtual bool Build();
			virtual bool CreatePipeline();
			virtual void DestroyPipeline();
			virtual void AttachToCommandBuffer(VkCommandBuffer & command_buffer);
		private:
			VkShaderModule m_shader_module;
		};
	}
}
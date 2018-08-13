#pragma once

#include <renderer\IComputePipeline.hpp>
#include <renderer\vulkan\VulkanPipeline.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanComputePipeline : public IComputePipeline, public VulkanPipeline
		{
		public:
			VulkanComputePipeline(const char* path, unsigned int x, unsigned int y, unsigned int z);
			virtual void AttachBuffer(IUniformBuffer* buffer);
			virtual void Build();
		private:

		};
	}
}
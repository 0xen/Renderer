#pragma once

#include <renderer\vulkan\VulkanHeader.hpp>
#include <renderer\IPipeline.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanPipeline : public virtual IPipeline
		{
		public:
			VulkanPipeline(const char* path);
			virtual void AttachBuffer(IUniformBuffer* buffer);
			virtual void Build();
		private:

		protected:
			VkPipeline * m_pipeline;
		};
	}
}
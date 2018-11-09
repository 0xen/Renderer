#pragma once

#include <renderer/IModel.hpp>

namespace Renderer
{

	namespace Vulkan
	{
		class VulkanModelPool;
		class VulkanModel : public IModel
		{
		public:
			VulkanModel(VulkanModelPool* pool, unsigned int model_pool_index);
			virtual void ShouldRender(bool render);
		private:
			VulkanModelPool * m_pool;
		};
	}
}
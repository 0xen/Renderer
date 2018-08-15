#pragma once

#include <renderer/IModel.hpp>

namespace Renderer
{

	namespace Vulkan
	{
		class VulkanModel : public IModel
		{
		public:
			VulkanModel(unsigned int model_pool_index);
		};
	}
}
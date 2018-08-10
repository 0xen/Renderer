#pragma once

#include <renderer/IRenderer.hpp>
#include <renderer/vulkan/VulkanInstance.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanRenderer : public IRenderer
		{
		public:
			VulkanRenderer();
			~VulkanRenderer();
			virtual bool Start();
			virtual void Update();
			virtual void Stop();
		private:
			VulkanInstance * m_instance;
		};
	}
}
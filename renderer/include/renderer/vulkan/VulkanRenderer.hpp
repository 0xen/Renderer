#pragma once

#include <renderer/IRenderer.hpp>
#include <renderer/vulkan/VulkanStatus.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanInstance;
		class VulkanPhysicalDevice;
		class VulkanRenderer : public IRenderer, public VulkanStatus
		{
		public:
			VulkanRenderer();
			~VulkanRenderer();
			virtual bool Start(Renderer::NativeWindowHandle window_handle);
			virtual void Update();
			virtual void Stop();
		private:
			void CreateSurface(Renderer::NativeWindowHandle window_handle);
			VulkanInstance * m_instance;
			VkSurfaceKHR m_surface;
			VulkanPhysicalDevice* m_physical_device;
		};
	}
}
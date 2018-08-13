#pragma once

#include <renderer/IRenderer.hpp>
#include <renderer/vulkan/VulkanStatus.hpp>
#include <renderer\DescriptorType.hpp>
#include <renderer\ShaderStage.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanInstance;
		class VulkanPhysicalDevice;
		class VulkanDevice;
		class VulkanSwapchain;
		class VulkanRenderer : public IRenderer, public VulkanStatus
		{
		public:
			using IRenderer::CreateUniformBuffer;
			VulkanRenderer();
			~VulkanRenderer();
			virtual bool Start(Renderer::NativeWindowHandle* window_handle);
			virtual void Update();
			virtual void Stop();
			virtual void Rebuild();
			virtual IUniformBuffer* CreateUniformBuffer(void* dataPtr, unsigned int indexSize, unsigned int elementCount, DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding);
		private:
			void CreateSurface(Renderer::NativeWindowHandle* window_handle);
			Renderer::NativeWindowHandle* m_window_handle;
			VulkanInstance * m_instance;
			VkSurfaceKHR m_surface;
			VulkanPhysicalDevice* m_physical_device;
			VulkanDevice* m_device;
			VulkanSwapchain* m_swapchain;
		};
	}
}
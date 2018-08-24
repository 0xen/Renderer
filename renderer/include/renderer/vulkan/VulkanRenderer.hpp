#pragma once

#include <renderer/vulkan/VulkanStatus.hpp>
#include <renderer/IRenderer.hpp>
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
			VulkanRenderer();
			~VulkanRenderer();
			virtual bool Start(Renderer::NativeWindowHandle* window_handle);
			virtual void Update();
			virtual void Stop();
			virtual void Rebuild();
			virtual IUniformBuffer* CreateUniformBuffer(void* dataPtr, unsigned int indexSize, unsigned int elementCount, ShaderStage shader_stage, unsigned int binding);

			virtual IVertexBuffer* CreateVertexBuffer(void* dataPtr, unsigned int indexSize, unsigned int elementCount);

			virtual IIndexBuffer* CreateIndexBuffer(void* dataPtr, unsigned int indexSize, unsigned int elementCount);

			virtual IGraphicsPipeline* CreateGraphicsPipeline(std::map<ShaderStage, const char*> paths);

			virtual IComputePipeline* CreateComputePipeline(const char* path, unsigned int x, unsigned int y, unsigned int z);

			virtual IComputeProgram* CreateComputeProgram();

			virtual IModelPool* CreateModelPool(IVertexBuffer* vertex_buffer, IIndexBuffer* index_buffer);

			virtual ITextureBuffer* CreateTextureBuffer(void* dataPtr, DataFormat format, unsigned int width, unsigned int height, unsigned int binding);
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
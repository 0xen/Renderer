#pragma once

#include <renderer/vulkan/VulkanStatus.hpp>
#include <renderer/IRenderer.hpp>
#include <renderer\DescriptorType.hpp>
#include <renderer\ShaderStage.hpp>

#include <imgui.h>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanInstance;
		class VulkanPhysicalDevice;
		class VulkanDevice;
		class VulkanSwapchain;
		class VulkanDescriptor;
		class VulkanGraphicsPipeline;
		class VulkanVertexBuffer;
		class VulkanIndexBuffer;
		class VulkanTextureBuffer;

		class VulkanRenderer : public IRenderer, public VulkanStatus
		{
		public:
			VulkanRenderer();
			~VulkanRenderer();
			virtual bool Start(Renderer::NativeWindowHandle* window_handle);
			virtual void InitilizeImGUI();
			virtual void RenderImGUI();
			virtual void Update();
			virtual void Stop();
			virtual void Rebuild();
			virtual IUniformBuffer* CreateUniformBuffer(void* dataPtr, unsigned int indexSize, unsigned int elementCount);

			virtual IVertexBuffer* CreateVertexBuffer(void* dataPtr, unsigned int indexSize, unsigned int elementCount);

			virtual IIndexBuffer* CreateIndexBuffer(void* dataPtr, unsigned int indexSize, unsigned int elementCount);

			virtual IGraphicsPipeline* CreateGraphicsPipeline(std::map<ShaderStage, const char*> paths);

			virtual IComputePipeline* CreateComputePipeline(const char* path, unsigned int x, unsigned int y, unsigned int z);

			virtual IComputeProgram* CreateComputeProgram();

			virtual IModelPool* CreateModelPool(IVertexBuffer* vertex_buffer, IIndexBuffer* index_buffer);

			virtual ITextureBuffer* CreateTextureBuffer(void* dataPtr, DataFormat format, unsigned int width, unsigned int height);

			virtual IDescriptor* CreateDescriptor(DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding);

			virtual IDescriptorPool* CreateDescriptorPool(std::vector<IDescriptor*> descriptors);

			static VkDescriptorType ToDescriptorType(DescriptorType descriptor_type);

			static VkShaderStageFlagBits ToVulkanShader(ShaderStage stage);
		private:
			void CreateSurface(Renderer::NativeWindowHandle* window_handle);
			Renderer::NativeWindowHandle* m_window_handle;
			VulkanInstance * m_instance;
			VkSurfaceKHR m_surface;
			VulkanPhysicalDevice* m_physical_device;
			VulkanDevice* m_device;
			VulkanSwapchain* m_swapchain;

			//ImGUI
			VulkanTextureBuffer* m_font_texture;
			VulkanGraphicsPipeline* m_imgui_pipeline;
			VkRenderPass m_render_pass;

			VulkanVertexBuffer* vertexBuffer = nullptr;
			VulkanIndexBuffer* indexBuffer = nullptr;

			ImDrawIdx* indexData = nullptr;
			ImDrawVert* vertexData = nullptr;
			std::vector<VkCommandBuffer> m_command_buffers;
			IDescriptorSet* texture_descriptor_set;
			IDescriptorPool* font_texture_pool;
		};
	}
}
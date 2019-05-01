#pragma once

#include <renderer\vulkan\VulkanGraphicsPipeline.hpp>
#include <renderer\vulkan\VulkanPipeline.hpp>


namespace Renderer
{
	namespace Vulkan
	{

		class VulkanSwapchain;
		class VulkanRaytracePipeline : public VulkanGraphicsPipeline
		{
		public:
			VulkanRaytracePipeline(VulkanDevice * device, VulkanSwapchain* swapchain, std::map<ShaderStage, const char*> paths);
			virtual ~VulkanRaytracePipeline();

			virtual bool Build();
			virtual bool CreatePipeline();
			virtual void DestroyPipeline();
			virtual void AttachToCommandBuffer(VkCommandBuffer & command_buffer);
			virtual void AttachModelPool(IModelPool* model_pool);
			virtual void AttachVertexBinding(VertexBase vertex_binding);
			virtual void UseDepth(bool depth);
			virtual void UseCulling(bool culling);
			virtual void DefinePrimitiveTopology(PrimitiveTopology top);
		private:
			VulkanDevice * m_device;
			std::vector<VkRayTracingShaderGroupCreateInfoNV> m_shader_groups;
			uint32_t m_shader_groupIndex = 0;
			bool m_recording_hit_shaders = false;
			std::vector<VkPipelineShaderStageCreateInfo> m_shader_stages;
			std::vector<VertexBase> m_vertex_bases;
			VulkanSwapchain * m_swapchain;
		};

	}
}
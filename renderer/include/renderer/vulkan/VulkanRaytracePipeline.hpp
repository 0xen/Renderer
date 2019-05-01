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
			VulkanRaytracePipeline(VulkanDevice * device, VulkanSwapchain* swapchain, std::map<ShaderStage, const char*> paths, std::vector<std::map<ShaderStage, const char*>> hitgroups);
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

			void SetMaxRecursionDepth(uint32_t max_depth);
		private:
			VulkanDevice * m_device;
			std::vector<std::map<ShaderStage, const char*>> m_hitgroups;
			std::vector<VkRayTracingShaderGroupCreateInfoNV> m_shader_groups;
			uint32_t m_shader_groupIndex = 0;
			std::vector<VkPipelineShaderStageCreateInfo> m_shader_stages;
			std::vector<VertexBase> m_vertex_bases;
			VulkanSwapchain * m_swapchain;
			uint32_t m_max_depth = 1;
		};

	}
}
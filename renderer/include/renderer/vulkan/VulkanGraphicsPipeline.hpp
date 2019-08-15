#pragma once

#include <renderer\vulkan\VulkanPipeline.hpp>
#include <renderer\vulkan\VulkanStatus.hpp>

#include <map>

namespace Renderer
{
	class VertexBase;
	namespace Vulkan
	{
		class VulkanGraphicsPipeline;

		struct VulkanGraphicsPipelineConfig
		{
			VulkanGraphicsPipeline* parent;
			std::vector<VkDynamicState> dynamic_states;
			bool use_depth_stencil;
			bool allow_darivatives;
			VkCullModeFlagBits culling;
			VkPrimitiveTopology topology;
			VkFrontFace front_face;
			VkPolygonMode polygon_mode;
			unsigned int subpass = 0;
		};

		class VulkanRenderPass;
		class VulkanModelPool;
		class VulkanGraphicsPipeline : public VulkanPipeline, public VulkanStatus
		{
		public:
			VulkanGraphicsPipeline(VulkanDevice * device, VulkanRenderPass* renderpass, std::vector<std::pair<VkShaderStageFlagBits, const char*>> paths);
			virtual ~VulkanGraphicsPipeline();
			virtual bool Build();
			virtual bool CreatePipeline();
			virtual void DestroyPipeline();
			virtual void AttachToCommandBuffer(VkCommandBuffer & command_buffer);
			virtual void AttachModelPool(VulkanModelPool* model_pool);
			virtual void AttachVertexBinding(VertexBase vertex_binding);
			bool HasChanged();
			VulkanGraphicsPipelineConfig& GetGraphicsPipelineConfig();

			virtual void AttachPipeline(VkCommandBuffer & command_buffer);
			virtual void BindDescriptorSets(VkCommandBuffer & command_buffer);
			virtual void RenderModels(VkCommandBuffer & command_buffer);

		private:


			void InitPipelineCreateInfo();

			VulkanGraphicsPipelineConfig m_graphics_pipeline_config;
			VulkanRenderPass * m_renderpass;
			std::vector<VkPipelineShaderStageCreateInfo> m_shader_stages;
			std::vector<VkVertexInputBindingDescription> m_binding_descriptions;
			std::vector<VkVertexInputAttributeDescription> m_attribute_descriptions;
			std::vector<VulkanModelPool*> m_model_pools;
			std::vector<VertexBase> m_vertex_bases;
			bool m_change;
		};
	}
}
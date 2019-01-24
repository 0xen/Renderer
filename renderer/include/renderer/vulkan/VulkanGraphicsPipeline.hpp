#pragma once

#include <renderer\IGraphicsPipeline.hpp>
#include <renderer\vulkan\VulkanPipeline.hpp>
#include <renderer\vulkan\VulkanStatus.hpp>
#include <renderer/VertexBase.hpp>
#include <renderer/DataFormat.hpp>

#include <map>

namespace Renderer
{
	namespace Vulkan
	{

		class VulkanSwapchain;
		class VulkanModelPool;
		class VulkanGraphicsPipeline : public IGraphicsPipeline, public VulkanPipeline, public VulkanStatus
		{
		public:
			VulkanGraphicsPipeline(VulkanDevice * device, VulkanSwapchain* swapchain, std::map<ShaderStage, const char*> paths);
			virtual ~VulkanGraphicsPipeline();
			virtual bool Build();
			virtual bool CreatePipeline();
			virtual void DestroyPipeline();
			virtual void AttachToCommandBuffer(VkCommandBuffer & command_buffer);
			virtual void AttachModelPool(IModelPool* model_pool);
			virtual void AttachVertexBinding(VertexBase vertex_binding);
			virtual void UseDepth(bool depth);
			virtual void UseCulling(bool culling);
			virtual void DefinePrimitiveTopology(PrimitiveTopology top);
			bool HasChanged();
		private:
			static VkShaderStageFlagBits GetShaderStageFlag(ShaderStage stage);
			static VkFormat GetFormat(Renderer::DataFormat format);
			static VkVertexInputRate GetVertexInputRate(Renderer::VertexInputRate input_rate);

			static std::map<Renderer::ShaderStage, VkShaderStageFlagBits> m_shader_stage_flags;
			static std::map<Renderer::DataFormat, VkFormat> m_formats;
			static std::map<Renderer::VertexInputRate, VkVertexInputRate> m_vertex_input_rates;


			VulkanSwapchain * m_swapchain;
			std::vector<VkPipelineShaderStageCreateInfo> m_shader_stages;
			std::vector<VkVertexInputBindingDescription> m_binding_descriptions;
			std::vector<VkVertexInputAttributeDescription> m_attribute_descriptions;
			std::vector<VulkanModelPool*> m_model_pools;
			std::vector<VertexBase> m_vertex_bases;
			VkPrimitiveTopology m_topology;
			bool m_change;
			bool m_use_depth_stencil = true;
			bool m_use_culling = false;
		};
	}
}
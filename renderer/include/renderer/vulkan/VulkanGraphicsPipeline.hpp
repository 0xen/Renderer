#pragma once

#include <renderer\vulkan\VulkanPipeline.hpp>
#include <renderer\vulkan\VulkanStatus.hpp>

#include <map>

namespace Renderer
{
	class VertexBase;
	namespace Vulkan
	{
		enum PrimitiveTopology
		{
			PointList,
			TriangleList
		};

		class VulkanSwapchain;
		class VulkanModelPool;
		class VulkanGraphicsPipeline : public VulkanPipeline, public VulkanStatus
		{
		public:
			VulkanGraphicsPipeline(VulkanDevice * device, VulkanSwapchain* swapchain, std::vector<std::pair<VkShaderStageFlagBits, const char*>> paths);
			virtual ~VulkanGraphicsPipeline();
			virtual bool Build();
			virtual bool CreatePipeline();
			virtual void DestroyPipeline();
			virtual void AttachToCommandBuffer(VkCommandBuffer & command_buffer);
			virtual void AttachModelPool(VulkanModelPool* model_pool);
			virtual void AttachVertexBinding(VertexBase vertex_binding);
			virtual void UseDepth(bool depth);
			virtual void UseCulling(bool culling);
			virtual void DefinePrimitiveTopology(PrimitiveTopology top);
			bool HasChanged();

		private:


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
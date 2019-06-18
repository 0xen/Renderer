#pragma once

#include <renderer\vulkan\VulkanGraphicsPipeline.hpp>
#include <renderer\vulkan\VulkanPipeline.hpp>
#include <renderer/vulkan/VulkanBufferData.hpp>


namespace Renderer
{
	namespace Vulkan
	{

		class VulkanSwapchain;

		struct SBTEntry
		{
			SBTEntry(uint32_t groupIndex, std::vector<unsigned char> inlineData) : m_groupIndex(groupIndex), m_inlineData(inlineData){}

			uint32_t                         m_groupIndex;
			const std::vector<unsigned char> m_inlineData;
		};

		class VulkanRaytracePipeline : public VulkanGraphicsPipeline
		{
		public:
			VulkanRaytracePipeline(VulkanDevice * device, VulkanSwapchain* swapchain, std::vector<std::pair<Renderer::ShaderStage, const char*>> paths, std::vector<std::map<ShaderStage, const char*>> hitgroups);
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

			void AddRayGenerationProgram(uint32_t group, const std::vector<unsigned char>& inlineData);
			void AddMissProgram(uint32_t group, const std::vector<unsigned char>& inlineData);
			void AddHitGroup(uint32_t group, const std::vector<unsigned char>& inlineData);


		private:

			VkDeviceSize GetEntrySize(std::vector<SBTEntry>);
			VkDeviceSize CopyShaderData(uint8_t* outputData, const std::vector<SBTEntry>& shaders, VkDeviceSize entrySize, const uint8_t* shaderHandleStorage);

			VkDeviceSize GetRayGenSectionSize();
			VkDeviceSize GetMissSectionSize();

			VulkanDevice * m_device;
			std::vector<std::map<ShaderStage, const char*>> m_hitgroups;
			std::vector<VkRayTracingShaderGroupCreateInfoNV> m_shader_groups;
			uint32_t m_shader_groupIndex = 0;
			std::vector<VkPipelineShaderStageCreateInfo> m_shader_stages;
			std::vector<VertexBase> m_vertex_bases;
			VulkanSwapchain * m_swapchain;
			uint32_t m_max_depth = 1;

			VulkanBufferData m_shaderBindingTable;

			VkDeviceSize m_rayGenEntrySize;
			VkDeviceSize m_missEntrySize;
			VkDeviceSize m_hitGroupEntrySize;

			// Ray generation shader entries
			std::vector<SBTEntry> m_rayGen;
			// Miss shader entries
			std::vector<SBTEntry> m_miss;
			// Hit group entries
			std::vector<SBTEntry> m_hitGroup;
		};

	}
}
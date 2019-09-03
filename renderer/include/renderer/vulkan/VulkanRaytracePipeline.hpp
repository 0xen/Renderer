#pragma once

#include <renderer\vulkan\VulkanGraphicsPipeline.hpp>
#include <renderer\vulkan\VulkanPipeline.hpp>
#include <renderer/vulkan/VulkanBufferData.hpp>


namespace Renderer
{
	class VertexBase;
	namespace Vulkan
	{

		class VulkanSwapchain;
		class VulkanRenderPass;
		struct SBTEntry
		{
			SBTEntry() {};
			SBTEntry(uint32_t groupIndex, std::vector<unsigned char> inlineData, const std::vector<unsigned int> constants) : m_groupIndex(groupIndex), m_inlineData(inlineData), m_constants(constants){}

			uint32_t                         m_groupIndex;
			const std::vector<unsigned char> m_inlineData;
			const std::vector<unsigned int> m_constants;
		};

		class VulkanRaytracePipeline : public VulkanGraphicsPipeline
		{
		public:
			VulkanRaytracePipeline(VulkanDevice * device, VulkanSwapchain * swapchain, VulkanRenderPass* renderpass, std::vector<std::pair<VkShaderStageFlagBits, const char*>> paths, std::vector<std::map<VkShaderStageFlagBits, const char*>> hitgroups);
			virtual ~VulkanRaytracePipeline();

			virtual bool Build();
			virtual bool CreatePipeline();
			virtual void DestroyPipeline();
			virtual void AttachToCommandBuffer(VkCommandBuffer & command_buffer);
			virtual void AttachVertexBinding(VertexBase vertex_binding);


			void SetMaxRecursionDepth(uint32_t max_depth);

			void AddRayGenerationProgram(uint32_t group, const std::vector<unsigned char>& inlineData, const std::vector<unsigned int>& constants);
			void AddMissProgram(uint32_t group, const std::vector<unsigned char>& inlineData, const std::vector<unsigned int>& constants);
			void AddHitGroup(uint32_t group, const std::vector<unsigned char>& inlineData, const std::vector<unsigned int>& constants);


		private:

			VkDeviceSize GetEntrySize(std::vector<SBTEntry>);
			VkDeviceSize CopyShaderData(uint8_t* outputData, const std::vector<SBTEntry>& shaders, VkDeviceSize entrySize, const uint8_t* shaderHandleStorage);

			VkDeviceSize GetRayGenSectionSize();
			VkDeviceSize GetMissSectionSize();

			VulkanDevice * m_device;
			std::vector<std::map<VkShaderStageFlagBits, const char*>> m_hitgroups;
			std::vector<VkRayTracingShaderGroupCreateInfoNV> m_shader_groups;
			uint32_t m_shader_groupIndex = 0;
			std::vector<VkPipelineShaderStageCreateInfo> m_shader_stages;
			std::vector<VertexBase> m_vertex_bases;
			VulkanSwapchain * m_swapchain;
			VulkanRenderPass* m_renderpass;
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
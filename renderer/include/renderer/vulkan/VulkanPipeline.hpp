#pragma once

#include <renderer\vulkan\VulkanHeader.hpp>

#include <vector>
#include <map>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanDevice;
		class VulkanUniformBuffer;
		class VulkanDescriptorPool;
		class VulkanDescriptorSet;
		class VulkanPipeline
		{
		public:
			VulkanPipeline(VulkanDevice * device, std::vector<std::pair<VkShaderStageFlagBits, const char*>> paths);
			~VulkanPipeline();
			virtual void AttachDescriptorPool(unsigned int setID, VulkanDescriptorPool* buffer);
			virtual void AttachDescriptorSet(unsigned int setID, VulkanDescriptorSet* descriptor_set);
			virtual bool Build();
			virtual bool CreatePipeline();
			virtual void DestroyPipeline();
			virtual void AttachToCommandBuffer(VkCommandBuffer & command_buffer);
			bool Rebuild();
			VkPipelineLayout& GetPipelineLayout();
			std::vector<std::pair<VkShaderStageFlagBits, const char*>> GetPaths();
		protected:
			VulkanDevice * m_device;

			VkPipelineLayout m_pipeline_layout;

			std::map<unsigned int, VulkanDescriptorPool*> m_descriptor_pools;
			std::map<unsigned int, VulkanDescriptorSet*> m_descriptor_sets;
			VkPipeline m_pipeline = VK_NULL_HANDLE;
		private:
			std::vector<std::pair<VkShaderStageFlagBits, const char*>> m_paths;
		};
	}
}
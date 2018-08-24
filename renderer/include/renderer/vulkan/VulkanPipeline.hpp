#pragma once

#include <renderer\vulkan\VulkanHeader.hpp>
#include <renderer\IPipeline.hpp>
#include <renderer\ShaderStage.hpp>

#include <vector>
#include <map>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanDevice;
		class VulkanBufferDescriptor;
		class VulkanUniformBuffer;
		class VulkanPipeline : public virtual IPipeline
		{
		public:
			VulkanPipeline(VulkanDevice * device, std::map<ShaderStage, const char*> paths);
			~VulkanPipeline();
			virtual void AttachBuffer(IUniformBuffer* buffer);
			virtual bool Build();
			virtual bool CreatePipeline();
			virtual void DestroyPipeline();
			virtual void AttachToCommandBuffer(VkCommandBuffer & command_buffer);
			bool Rebuild();
		protected:
			VulkanDevice * m_device;
			std::vector<VkDescriptorSetLayoutBinding> m_layout_bindings;
			std::vector<VkDescriptorPoolSize> m_descriptor_pool_sizes;
			std::vector<VkDescriptorSetLayout> m_descriptor_set_layouts;
			std::vector<VkWriteDescriptorSet> m_write_descriptor_sets;

			VkDescriptorPool m_descriptor_pool;
			VkDescriptorSetLayout m_descriptor_set_layout;
			VkDescriptorSet m_descriptor_set;
			VkPipelineLayout m_pipeline_layout;

			std::vector<VulkanBufferDescriptor*> m_descriptor;
			VkPipeline m_pipeline = VK_NULL_HANDLE;
		private:
		};
	}
}
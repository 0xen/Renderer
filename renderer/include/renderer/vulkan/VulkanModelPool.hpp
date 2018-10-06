#pragma once

#include <renderer/IModelPool.hpp>
#include <renderer/vulkan/VulkanModel.hpp>
#include <renderer/vulkan/VulkanUniformBuffer.hpp>

#include <glm/glm.hpp>
#include <map>
#include <vector>

namespace Renderer
{

	namespace Vulkan
	{
		class VulkanDevice;
		class VulkanDescriptorSet;
		class VulkanPipeline;
		class VulkanModelPool : public IModelPool
		{
		public:
			VulkanModelPool(VulkanDevice* device, IVertexBuffer* vertex_buffer);
			VulkanModelPool(VulkanDevice* device, IVertexBuffer* vertex_buffer, IIndexBuffer* index_buffer);
			~VulkanModelPool();
			virtual IModel * CreateModel();
			virtual void AttachBuffer(unsigned int index, IUniformBuffer * buffer);
			virtual void AttachDescriptorSet(unsigned int index, IDescriptorSet* descriptor_set);
			void AttachToCommandBuffer(VkCommandBuffer & command_buffer, VulkanPipeline* pipeline);
			bool HasChanged();
		private:
			unsigned int m_current_index;
			VulkanDevice * m_device;
			std::map<unsigned int, VulkanDescriptorSet*> m_descriptor_sets;
			std::map<unsigned int, VulkanUniformBuffer*> m_buffers;
			std::map<unsigned int, VulkanModel*> m_models;
			VulkanBuffer* m_indirect_draw_buffer;
			VkDrawIndirectCommand m_vertex_indirect_command;
			VkDrawIndexedIndirectCommand m_indexed_indirect_command;
			bool m_change;
		};
	}
}
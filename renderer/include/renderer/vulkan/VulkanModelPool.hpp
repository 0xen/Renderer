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
		class VulkanModelPool : public IModelPool
		{
		public:
			VulkanModelPool(VulkanDevice* device,IVertexBuffer* vertex_buffer, IIndexBuffer* index_buffer);
			~VulkanModelPool();
			virtual IModel * CreateModel();
			virtual void AttachBuffer(unsigned int index, IUniformBuffer * buffer);
			void AttachToCommandBuffer(VkCommandBuffer & command_buffer);
			bool HasChanged();
		private:
			unsigned int m_current_index;
			VulkanDevice * m_device;
			std::map<unsigned int, VulkanUniformBuffer*> m_buffers;
			std::map<unsigned int, VulkanModel*> m_models;
			VulkanBuffer* m_indirect_draw_buffer;
			VkDrawIndexedIndirectCommand m_indirect_command;
			bool m_change;
		};
	}
}
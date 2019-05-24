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
			virtual ~VulkanModelPool();
			virtual IModel * CreateModel();
			virtual IModel* GetModel(int index);
			virtual void RemoveModel(IModel* model);
			virtual void Update();
			virtual void AttachBuffer(unsigned int index, IUniformBuffer * buffer);
			virtual void UpdateModelBuffer(unsigned int index);
			virtual void AttachDescriptorSet(unsigned int index, IDescriptorSet* descriptor_set);
			virtual std::vector<IDescriptorSet*> GetDescriptorSets();
			virtual void SetVertexDrawCount(unsigned int count);
			virtual unsigned int GetLargestIndex();
			std::map<unsigned int, VulkanModel*>& GetModels();
			std::map<unsigned int, VulkanUniformBuffer*>& GetBuffers();
			void AttachToCommandBuffer(VkCommandBuffer & command_buffer, VulkanPipeline* pipeline);
			bool HasChanged();
		private:
			void ResizeIndirectArray(unsigned int size);
			void Render(unsigned int index, bool should_render);
			unsigned int m_current_index;
			unsigned int m_largest_index;
			std::vector<unsigned int> m_free_indexs;
			unsigned int m_vertex_draw_count;
			VulkanDevice * m_device;
			std::map<unsigned int, VulkanDescriptorSet*> m_descriptor_sets;
			std::map<unsigned int, VulkanUniformBuffer*> m_buffers;
			std::map<unsigned int, VulkanModel*> m_models;
			static const unsigned int m_indirect_array_padding;
			VulkanBuffer* m_indirect_draw_buffer = nullptr;



			std::vector<VkDrawIndirectCommand> m_vertex_indirect_command;
			std::vector<VkDrawIndexedIndirectCommand> m_indexed_indirect_command;

			/*union
			{
				//void* m_indirect_command;
				//VkDrawIndirectCommand* m_vertex_indirect_command;
				//VkDrawIndexedIndirectCommand* m_indexed_indirect_command;
			};*/

			bool m_change;

			friend VulkanModel;
		};
	}
}
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
			VulkanModelPool(VulkanDevice* device, IVertexBuffer* vertex_buffer, unsigned int vertex_offset, unsigned int vertex_size);
			VulkanModelPool(VulkanDevice* device, IVertexBuffer* vertex_buffer, unsigned int vertex_offset, unsigned int vertex_size, IIndexBuffer* index_buffer, unsigned int index_offset, unsigned int index_size);
			virtual ~VulkanModelPool();
			virtual IModel * CreateModel();
			virtual IModel* GetModel(int index);
			virtual void RemoveModel(IModel* model);
			virtual void Update();
			virtual void AttachBufferPool(unsigned int index, IBufferPool * buffer);
			virtual void UpdateModelBuffer(unsigned int index);
			virtual void AttachDescriptorSet(unsigned int index, IDescriptorSet* descriptor_set);
			virtual std::vector<IDescriptorSet*> GetDescriptorSets();
			virtual unsigned int GetLargestIndex();
			unsigned int GetModelBufferOffset(IModel* model,unsigned int buffer);
			std::map<unsigned int, VulkanModel*>& GetModels();
			std::map<unsigned int, IBufferPool*>& GetBufferPools();
			void AttachToCommandBuffer(VkCommandBuffer & command_buffer, VulkanPipeline* pipeline);
			bool HasChanged();
		private:
			void ResizeIndirectArray(unsigned int size);
			void Render(unsigned int index, bool should_render);
			unsigned int m_current_index;
			unsigned int m_largest_index;
			std::vector<unsigned int> m_free_indexs;
			VulkanDevice * m_device;
			std::map<unsigned int, VulkanDescriptorSet*> m_descriptor_sets;
			std::map<unsigned int, IBufferPool*> m_buffers;
			std::map<unsigned int, VulkanModel*> m_models;
			// Key: Model ID, Key2 BufferID, Value buffer offset
			std::map<unsigned int, std::map<unsigned int, unsigned int>> m_model_buffer_mapping;
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
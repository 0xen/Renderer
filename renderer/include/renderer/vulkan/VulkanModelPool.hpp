#pragma once

#include <renderer/vulkan/VulkanUniformBuffer.hpp>

#include <glm/glm.hpp>
#include <map>
#include <vector>

namespace Renderer
{

	namespace Vulkan
	{
		class VulkanModel;
		class VulkanDevice;
		class VulkanDescriptorSet;
		class VulkanPipeline;
		class VulkanVertexBuffer;
		class VulkanIndexBuffer;
		class VulkanBufferPool;

		enum ModelPoolUsage
		{
			SingleMesh,
			MultiMesh
		};

		class VulkanModelPool
		{
		public:
			VulkanModelPool(VulkanDevice* device, VulkanVertexBuffer* vertex_buffer, unsigned int vertex_offset, unsigned int vertex_size, ModelPoolUsage usage = SingleMesh);
			VulkanModelPool(VulkanDevice* device, VulkanVertexBuffer* vertex_buffer, unsigned int vertex_offset, unsigned int vertex_size, VulkanIndexBuffer* index_buffer, unsigned int index_offset, unsigned int index_size, ModelPoolUsage usage = SingleMesh);
			~VulkanModelPool();
			bool Indexed();
			VulkanModel * CreateModel();
			VulkanModel * CreateModel(unsigned int vertexOffset, unsigned int indexOffset, unsigned int indexSize);
			VulkanModel* GetModel(int index);
			void RemoveModel(VulkanModel* model);
			void Update();
			void AttachBufferPool(unsigned int index, VulkanBufferPool * buffer);
			void UpdateModelBuffer(unsigned int index);
			void AttachDescriptorSet(unsigned int index, VulkanDescriptorSet* descriptor_set);
			std::vector<VulkanDescriptorSet*> GetDescriptorSets();
			unsigned int GetLargestIndex();


			void SetVertexBuffer(VulkanVertexBuffer* vertex_buffer);
			VulkanVertexBuffer * GetVertexBuffer();
			VulkanIndexBuffer * GetIndexBuffer();
			VulkanBuffer* GetIndirectDrawBuffer();
			unsigned int GetVertexOffset();
			unsigned int GetIndexOffset();
			unsigned int GetVertexSize();
			unsigned int GetIndexSize();
			void SetVertexOffset(unsigned int offset);
			void SetIndexOffset(unsigned int offset);
			void SetVertexSize(unsigned int size);
			void SetIndexSize(unsigned int size);

			unsigned int GetModelBufferOffset(VulkanModel* model,unsigned int buffer);
			std::map<unsigned int, VulkanModel*>& GetModels();
			std::map<unsigned int, VulkanBufferPool*>& GetBufferPools();
			void AttachToCommandBuffer(VkCommandBuffer & command_buffer, VulkanPipeline* pipeline);
			bool HasChanged();
		private:
			void SetModelOffsets(unsigned int id, unsigned int vertexOffset, unsigned int indexOffset, unsigned int indexSize);
			void ResizeIndirectArray(unsigned int size);
			void Render(unsigned int index, bool should_render);
			unsigned int m_current_index;
			unsigned int m_largest_index;
			std::vector<unsigned int> m_free_indexs;
			VulkanDevice * m_device;
			std::map<unsigned int, VulkanDescriptorSet*> m_descriptor_sets;
			std::map<unsigned int, VulkanBufferPool*> m_buffers;
			std::map<unsigned int, VulkanModel*> m_models;
			// Key: Model ID, Key2 BufferID, Value buffer offset
			std::map<unsigned int, std::map<unsigned int, unsigned int>> m_model_buffer_mapping;
			static const unsigned int m_indirect_array_padding;
			VulkanBuffer* m_indirect_draw_buffer = nullptr;


			bool m_indexed;
			VulkanVertexBuffer * m_vertex_buffer;
			VulkanIndexBuffer * m_index_buffer;
			unsigned int m_vertex_offset;
			unsigned int m_vertex_size;
			unsigned int m_index_offset;
			unsigned int m_index_size;
			bool m_vertex_index_change = false;
			ModelPoolUsage m_model_pool_usage;


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
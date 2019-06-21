#pragma once

#include <vector>

namespace Renderer
{
	class IVertexBuffer;
	class IIndexBuffer;
	class IUniformBuffer;
	class IDescriptorSet;
	class IModel;
	class IBufferPool;
	class IModelPool
	{
	public:
		IModelPool(IVertexBuffer* vertex_buffer, unsigned int vertex_offset,unsigned int vertex_size);
		IModelPool(IVertexBuffer* vertex_buffer, unsigned int vertex_offset, unsigned int vertex_size, IIndexBuffer* index_buffer, unsigned int index_offset, unsigned int index_size);
		virtual ~IModelPool() {};
		bool Indexed();
		virtual IModel * CreateModel() = 0;
		virtual IModel* GetModel(int index) = 0;
		virtual void RemoveModel(IModel* model) = 0;
		virtual void Update() = 0;
		virtual void AttachBufferPool(unsigned int index, IBufferPool * buffer) = 0;
		virtual void UpdateModelBuffer(unsigned int index) = 0;
		virtual void AttachDescriptorSet(unsigned int index, IDescriptorSet* descriptor_set) = 0;
		virtual std::vector<IDescriptorSet*> GetDescriptorSets() = 0;
		virtual void SetVertexDrawCount(unsigned int count) = 0;
		void SetVertexBuffer(IVertexBuffer* vertex_buffer);
		IVertexBuffer * GetVertexBuffer();
		IIndexBuffer * GetIndexBuffer();
		unsigned int GetVertexOffset();
		unsigned int GetIndexOffset();
		unsigned int GetVertexSize();
		unsigned int GetIndexSize();
		virtual unsigned int GetLargestIndex()= 0;
	protected:
		bool m_indexed;
		IVertexBuffer * m_vertex_buffer;
		IIndexBuffer * m_index_buffer;
		unsigned int m_vertex_offset;
		unsigned int m_vertex_size;
		unsigned int m_index_offset;
		unsigned int m_index_size;
	};
}
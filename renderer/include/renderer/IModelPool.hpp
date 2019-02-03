#pragma once

namespace Renderer
{
	class IVertexBuffer;
	class IIndexBuffer;
	class IUniformBuffer;
	class IDescriptorSet;
	class IModel;
	class IModelPool
	{
	public:
		IModelPool(IVertexBuffer* vertex_buffer);
		IModelPool(IVertexBuffer* vertex_buffer, IIndexBuffer* index_buffer);
		virtual ~IModelPool() {};
		bool Indexed();
		virtual IModel * CreateModel() = 0;
		virtual void RemoveModel(IModel* model) = 0;
		virtual void Update() = 0;
		virtual void AttachBuffer(unsigned int index, IUniformBuffer * buffer) = 0;
		virtual void AttachDescriptorSet(unsigned int index, IDescriptorSet* descriptor_set) = 0;
		virtual void SetVertexDrawCount(unsigned int count) = 0;
		void SetVertexBuffer(IVertexBuffer* vertex_buffer);
		IVertexBuffer * GetVertexBuffer();
		IIndexBuffer * GetIndexBuffer();
	protected:
		bool m_indexed;
		IVertexBuffer * m_vertex_buffer;
		IIndexBuffer * m_index_buffer;
	};
}
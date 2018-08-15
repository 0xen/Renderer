#include <renderer/IModelPool.hpp>

Renderer::IModelPool::IModelPool(IVertexBuffer * vertex_buffer, IIndexBuffer * index_buffer)
{
	m_vertex_buffer = vertex_buffer;
	m_index_buffer = index_buffer;
}

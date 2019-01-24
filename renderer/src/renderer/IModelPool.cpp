#include <renderer/IModelPool.hpp>

Renderer::IModelPool::IModelPool(IVertexBuffer * vertex_buffer)
{
	m_vertex_buffer = vertex_buffer;
	m_indexed = false;
}

Renderer::IModelPool::IModelPool(IVertexBuffer * vertex_buffer, IIndexBuffer * index_buffer)
{
	m_vertex_buffer = vertex_buffer;
	m_index_buffer = index_buffer;
	m_indexed = true;
}

bool Renderer::IModelPool::Indexed()
{
	return m_indexed;
}

void Renderer::IModelPool::SetVertexBuffer(IVertexBuffer * vertex_buffer)
{
	m_vertex_buffer = vertex_buffer;
}

Renderer::IVertexBuffer * Renderer::IModelPool::GetVertexBuffer()
{
	return m_vertex_buffer;
}

Renderer::IIndexBuffer * Renderer::IModelPool::GetIndexBuffer()
{
	return m_index_buffer;
}

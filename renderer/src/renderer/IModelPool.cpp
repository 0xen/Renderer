#include <renderer/IModelPool.hpp>

Renderer::IModelPool::IModelPool(IVertexBuffer * vertex_buffer, unsigned int vertex_offset, unsigned int vertex_size) : m_vertex_offset(vertex_offset) , m_vertex_size(vertex_size)
{
	m_vertex_buffer = vertex_buffer;
	m_indexed = false;
}

Renderer::IModelPool::IModelPool(IVertexBuffer * vertex_buffer, unsigned int vertex_offset, unsigned int vertex_size, IIndexBuffer * index_buffer, unsigned int index_offset, unsigned int index_size) : m_vertex_offset(vertex_offset), m_vertex_size(vertex_size), m_index_offset(index_offset), m_index_size(index_size)
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

unsigned int Renderer::IModelPool::GetVertexOffset()
{
	return m_vertex_offset;
}

unsigned int Renderer::IModelPool::GetIndexOffset()
{
	return m_index_offset;
}

unsigned int Renderer::IModelPool::GetVertexSize()
{
	return m_vertex_size;
}

unsigned int Renderer::IModelPool::GetIndexSize()
{
	return m_index_size;
}

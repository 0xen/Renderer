#include <renderer/vulkan/VulkanModel.hpp>
#include <renderer\vulkan\VulkanModelPool.hpp>

Renderer::Vulkan::VulkanModel::VulkanModel(VulkanModelPool* pool, unsigned int model_pool_index) :
	m_model_pool_index(model_pool_index), m_pool(pool)
{
	m_rendering = false;
}

Renderer::Vulkan::VulkanModel::VulkanModel(VulkanModelPool* pool, unsigned int model_pool_index, unsigned int vertexOffset, unsigned int indexOffset, unsigned int indexSize) :
	m_model_pool_index(model_pool_index), m_pool(pool), m_vertexOffset(vertexOffset), m_indexOffset(indexOffset), m_indexSize(indexSize)
{
	m_rendering = false;
}

void Renderer::Vulkan::VulkanModel::SetDataPointer(unsigned int index, void * data)
{
	m_data_pointers[index] = data;
}

void Renderer::Vulkan::VulkanModel::SetData(unsigned int index, void * data, unsigned int size)
{
	memcpy(m_data_pointers[index], data, size);
}

unsigned int Renderer::Vulkan::VulkanModel::GetModelPoolIndex()
{
	return m_model_pool_index;
}

void Renderer::Vulkan::VulkanModel::Remove()
{
	GetModelPool()->RemoveModel(this);
}


void Renderer::Vulkan::VulkanModel::ShouldRender(bool render)
{
	m_pool->Render(m_model_pool_index, render);
	m_rendering = render;
}

bool Renderer::Vulkan::VulkanModel::Rendering()
{
	return m_rendering;
}

Renderer::Vulkan::VulkanModelPool * Renderer::Vulkan::VulkanModel::GetModelPool()
{
	return m_pool;
}

unsigned int Renderer::Vulkan::VulkanModel::GetVertexOffset()
{
	return m_vertexOffset;
}

unsigned int Renderer::Vulkan::VulkanModel::GetIndexOffset()
{
	return m_indexOffset;
}

unsigned int Renderer::Vulkan::VulkanModel::GetIndexSize()
{
	return m_indexSize;
}

void Renderer::Vulkan::VulkanModel::SetVertexOffset(unsigned int offset)
{
	m_vertexOffset = offset;
	m_pool->m_vertex_index_change = true;
}

void Renderer::Vulkan::VulkanModel::SetIndexOffset(unsigned int offset)
{
	m_indexOffset = offset;
	m_pool->m_vertex_index_change = true;
}

void Renderer::Vulkan::VulkanModel::SetIndexSize(unsigned int size)
{
	m_indexSize = size;
	m_pool->m_vertex_index_change = true;
}

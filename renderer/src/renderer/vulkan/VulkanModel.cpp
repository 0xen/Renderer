#include <renderer/vulkan/VulkanModel.hpp>
#include <renderer\vulkan\VulkanModelPool.hpp>

Renderer::Vulkan::VulkanModel::VulkanModel(VulkanModelPool* pool, unsigned int model_pool_index) :
	m_model_pool_index(model_pool_index), m_pool(pool)
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

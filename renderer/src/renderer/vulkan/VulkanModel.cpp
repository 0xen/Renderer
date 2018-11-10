#include <renderer/vulkan/VulkanModel.hpp>
#include <renderer\vulkan\VulkanModelPool.hpp>

Renderer::Vulkan::VulkanModel::VulkanModel(VulkanModelPool* pool, unsigned int model_pool_index) :
	IModel(model_pool_index), m_pool(pool)
{
	m_rendering = false;
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

#include <renderer/IRenderer.hpp>

#include <renderer\vulkan\VulkanRenderer.hpp>

using namespace Renderer;

std::vector<IRenderer*> IRenderer::m_renderers;

Renderer::IRenderer::IRenderer()
{
}

IRenderer * Renderer::IRenderer::CreateRenderer(const RenderingAPI api)
{
	IRenderer * renderer = nullptr;
	switch (api)
	{
	case Vulkan:
		renderer = new VulkanRenderer();
		break;
	}
	if (renderer != nullptr) m_renderers.push_back(renderer);
	return renderer;
}

void Renderer::IRenderer::UpdateAll()
{
	for (int i = 0; i < m_renderers.size(); i++)
	{
		m_renderers[i]->Update();
	}
}

void Renderer::IRenderer::UnregisterRenderer(IRenderer * renderer)
{
	auto it = std::find(m_renderers.begin(), m_renderers.end(), renderer);
	if (it != m_renderers.end())
		m_renderers.erase(it);
}

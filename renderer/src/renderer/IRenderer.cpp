#include <renderer/IRenderer.hpp>

#include <renderer\vulkan\VulkanRenderer.hpp>

using namespace Renderer;

IRenderer * Renderer::IRenderer::CreateRenderer(const RenderingAPI api)
{
	switch (api)
	{
	case Vulkan:
		return new VulkanRenderer();
		break;
	}
	return nullptr;
}

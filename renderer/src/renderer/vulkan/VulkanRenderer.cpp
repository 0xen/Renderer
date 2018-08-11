#include <renderer/vulkan/VulkanRenderer.hpp>

#include <assert.h>

using namespace Renderer::Vulkan;
VulkanRenderer::VulkanRenderer() : IRenderer()
{
}

VulkanRenderer::~VulkanRenderer()
{
	Stop();
}

bool VulkanRenderer::Start()
{
	m_instance = new VulkanInstance();

	Status::ErrorCheck(m_instance);

	if (HasError())return false;

	return true;
}

void VulkanRenderer::Update()
{
}

void VulkanRenderer::Stop()
{
	delete m_instance;
}

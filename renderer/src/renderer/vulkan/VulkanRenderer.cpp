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

	assert(!m_instance->HasError() && "Unable to create a vulkan instance");

	return true;
}

void VulkanRenderer::Update()
{
}

void VulkanRenderer::Stop()
{
	delete m_instance;
}

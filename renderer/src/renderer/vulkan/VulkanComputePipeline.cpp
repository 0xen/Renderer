#include <renderer/vulkan/VulkanComputePipeline.hpp>

Renderer::Vulkan::VulkanComputePipeline::VulkanComputePipeline(const char * path, unsigned int x, unsigned int y, unsigned int z) :
	IComputePipeline(path,x,y,z),
	VulkanPipeline(path),
	IPipeline(path)
{

}

void Renderer::Vulkan::VulkanComputePipeline::AttachBuffer(IUniformBuffer * buffer)
{
}

void Renderer::Vulkan::VulkanComputePipeline::Build()
{
}

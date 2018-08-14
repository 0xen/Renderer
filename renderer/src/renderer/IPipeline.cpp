#include <renderer/IPipeline.hpp>

Renderer::IPipeline::IPipeline()
{
}

Renderer::IPipeline::IPipeline(std::map<ShaderStage, const char*> paths)
{
	m_paths = paths;
}

std::map<Renderer::ShaderStage, const char*> Renderer::IPipeline::GetPaths()
{
	return m_paths;
}
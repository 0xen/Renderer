#include <renderer/IPipeline.hpp>

Renderer::IPipeline::IPipeline()
{
}

Renderer::IPipeline::IPipeline(std::vector<std::pair<Renderer::ShaderStage, const char*>> paths)
{
	m_paths = paths;
}

std::vector<std::pair<Renderer::ShaderStage, const char*>> Renderer::IPipeline::GetPaths()
{
	return m_paths;
}
#include <renderer/IPipeline.hpp>

Renderer::IPipeline::IPipeline()
{
}

Renderer::IPipeline::IPipeline(const char * path)
{
	m_path = path;
}

const char * Renderer::IPipeline::GetPath()
{
	return m_path;
}
#include <renderer/IComputePipeline.hpp>

Renderer::IComputePipeline::IComputePipeline(const char * path, unsigned int x, unsigned int y, unsigned int z) :
	IPipeline(path)
{
	m_x = x;
	m_y = y;
	m_z = z;
}

unsigned int Renderer::IComputePipeline::GetX()
{
	return m_x;
}

unsigned int Renderer::IComputePipeline::GetY()
{
	return m_y;
}

unsigned int Renderer::IComputePipeline::GetZ()
{
	return m_z;
}

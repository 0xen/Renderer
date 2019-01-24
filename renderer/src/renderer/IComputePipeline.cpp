#include <renderer/IComputePipeline.hpp>
#include <renderer/ShaderStage.hpp>
#include <map>

Renderer::IComputePipeline::IComputePipeline(const char * path, unsigned int x, unsigned int y, unsigned int z) :
	IPipeline({ { ShaderStage::COMPUTE_SHADER, path } })
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

void Renderer::IComputePipeline::SetX(unsigned int x)
{
	m_x = x;
}

void Renderer::IComputePipeline::SetY(unsigned int y)
{
	m_y = y;
}

void Renderer::IComputePipeline::SetZ(unsigned int z)
{
	m_z = z;
}

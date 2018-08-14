#include <renderer/IComputeProgram.hpp>

Renderer::IComputeProgram::IComputeProgram()
{
}

void Renderer::IComputeProgram::AttachPipeline(IComputePipeline * pipeline)
{
	m_pipelines.push_back(pipeline);
}

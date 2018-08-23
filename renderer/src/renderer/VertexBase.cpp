#include <renderer/VertexBase.hpp>

using namespace Renderer;

void Renderer::VertexBase::SetBinding(unsigned int binding)
{
	m_binding = binding;
}

void Renderer::VertexBase::SetVertexInputRate(VertexInputRate input_rate)
{
	m_vertex_input_rate = input_rate;
}

void Renderer::VertexBase::SetSize(unsigned int size)
{
	m_size = size;
}

void Renderer::VertexBase::AddVertexBinding(VertexBinding binding)
{
	m_vertex_bindings.push_back(binding);
}

VertexInputRate Renderer::VertexBase::GetVertexInputRate()
{
	return m_vertex_input_rate;
}

unsigned int Renderer::VertexBase::GetSize()
{
	return m_size;
}

unsigned int Renderer::VertexBase::GetBinding()
{
	return m_binding;
}

std::vector<VertexBinding>& Renderer::VertexBase::GetBindings()
{
	return m_vertex_bindings;
}

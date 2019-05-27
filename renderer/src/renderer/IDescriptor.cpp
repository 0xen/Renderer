#include <renderer/IDescriptor.hpp>

using namespace Renderer;

Renderer::IDescriptor::IDescriptor()
{
}

Renderer::IDescriptor::IDescriptor(unsigned int binding, unsigned int count)
{
	m_binding = binding;
	m_count = count;
}

Renderer::IDescriptor::IDescriptor(DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding, unsigned int count)
{
	m_descriptor_type = descriptor_type;
	m_shader_stage = shader_stage;
	m_binding = binding;
	m_count = count;
}

ShaderStage Renderer::IDescriptor::GetShaderStage()
{
	return m_shader_stage;
}

DescriptorType Renderer::IDescriptor::GetDescriptorType()
{
	return m_descriptor_type;
}

unsigned int Renderer::IDescriptor::GetBinding()
{
	return m_binding;
}

unsigned int Renderer::IDescriptor::GetCount()
{
	return m_count;
}

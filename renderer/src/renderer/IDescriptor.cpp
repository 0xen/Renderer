#include <renderer/IDescriptor.hpp>

using namespace Renderer;

Renderer::IDescriptor::IDescriptor()
{
}

Renderer::IDescriptor::IDescriptor(DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding)
{
	m_descriptor_type = descriptor_type;
	m_shader_stage = shader_stage;
	m_binding = binding;
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

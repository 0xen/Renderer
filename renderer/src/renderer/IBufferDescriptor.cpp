#include <renderer/IBufferDescriptor.hpp>

Renderer::IBufferDescriptor::IBufferDescriptor() {}

Renderer::IBufferDescriptor::IBufferDescriptor(DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding)
{
	m_descriptor_type = descriptor_type;
	m_shader_stage = shader_stage;
	m_binding = binding;
}

Renderer::IBufferDescriptor::~IBufferDescriptor()
{
}

Renderer::ShaderStage Renderer::IBufferDescriptor::GetShaderStage()
{
	return m_shader_stage;
}

Renderer::DescriptorType Renderer::IBufferDescriptor::GetDescriptorType()
{
	return m_descriptor_type;
}

unsigned int Renderer::IBufferDescriptor::GetBinding()
{
	return m_binding;
}

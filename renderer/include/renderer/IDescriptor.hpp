#pragma once

#include <renderer\DescriptorType.hpp>
#include <renderer\ShaderStage.hpp>

namespace Renderer
{
	class IDescriptor
	{
	public:
		IDescriptor();
		IDescriptor(DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding);
		ShaderStage GetShaderStage();
		DescriptorType GetDescriptorType();
		unsigned int GetBinding();
	private:
		DescriptorType m_descriptor_type;
		ShaderStage m_shader_stage;
		unsigned int m_binding;
	};
}
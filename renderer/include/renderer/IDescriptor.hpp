#pragma once

#include <renderer\DescriptorType.hpp>
#include <renderer\ShaderStage.hpp>

namespace Renderer
{
	class IDescriptor
	{
	public:
		IDescriptor();
		IDescriptor(unsigned int binding,unsigned int count);
		IDescriptor(DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding, unsigned int count);
		ShaderStage GetShaderStage();
		DescriptorType GetDescriptorType();
		unsigned int GetBinding();
		unsigned int GetCount();
	private:
		DescriptorType m_descriptor_type;
		ShaderStage m_shader_stage;
		unsigned int m_binding;
		unsigned int m_count;
	};
}
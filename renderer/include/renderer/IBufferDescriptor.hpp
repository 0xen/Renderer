#pragma once

#include <renderer\DescriptorType.hpp>
#include <renderer\ShaderStage.hpp>

namespace Renderer
{
	class IBufferDescriptor
	{
	public:
		IBufferDescriptor();
		IBufferDescriptor(DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding);
		~IBufferDescriptor();
		ShaderStage GetShaderStage();
		DescriptorType GetDescriptorType();
		unsigned int GetBinding();
	private:
		DescriptorType m_descriptor_type;
		ShaderStage m_shader_stage;
		unsigned int m_binding;



	};
}
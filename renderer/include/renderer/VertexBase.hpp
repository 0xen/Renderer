#pragma once

#include <renderer\VertexBinding.hpp>
#include <renderer\VertexInputRate.hpp>

#include <vector>
#include <iostream>

namespace Renderer
{
	class VertexBase
	{
	public:
		VertexBase(VkVertexInputRate vertex_input_rate, std::vector<VertexBinding> vertex_bindings, unsigned int size, unsigned int binding) :
			vertex_input_rate(vertex_input_rate), vertex_bindings(vertex_bindings), size(size), binding(binding){}
		VkVertexInputRate vertex_input_rate;
		std::vector<VertexBinding> vertex_bindings;
		unsigned int size;
		unsigned int binding;
	};
}
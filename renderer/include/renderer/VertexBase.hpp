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
		void SetBinding(unsigned int binding);
		void SetVertexInputRate(VertexInputRate input_rate);
		void SetSize(unsigned int size);
		void AddVertexBinding(VertexBinding binding);
		VertexInputRate GetVertexInputRate();
		unsigned int GetSize();
		unsigned int GetBinding();
		std::vector<VertexBinding>& GetBindings();
	private:
		VertexInputRate m_vertex_input_rate;
		std::vector<VertexBinding> m_vertex_bindings;
		unsigned int m_size;
		unsigned int m_binding;
	};
}
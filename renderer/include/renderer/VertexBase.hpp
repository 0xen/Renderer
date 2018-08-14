#pragma once

#include <renderer\VertexBinding.hpp>

#include <vector>
#include <iostream>

namespace Renderer
{
	class VertexBase
	{
	public:
		static std::vector<VertexBinding> vertex_bindings;
		static unsigned int size;
	};
}
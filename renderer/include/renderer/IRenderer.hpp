#pragma once

#include <renderer/APIs.hpp>

namespace Renderer
{
	class IRenderer
	{
	public:
		static IRenderer* CreateRenderer(const RenderingAPI api);
	};
}
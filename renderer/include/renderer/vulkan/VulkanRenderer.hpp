#pragma once

#include <renderer/IRenderer.hpp>

namespace Renderer
{
	class VulkanRenderer : public IRenderer
	{
	public:
		VulkanRenderer();
		~VulkanRenderer();
		virtual bool Start();
		virtual void Update();
		virtual void Stop();
	};
}
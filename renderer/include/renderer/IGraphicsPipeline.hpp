#pragma once

#include <renderer\IPipeline.hpp>
#include <renderer\ShaderStage.hpp>

#include <map>

namespace Renderer
{
	class IGraphicsPipeline : public virtual IPipeline
	{
	public:
		IGraphicsPipeline(std::map<ShaderStage,const char*> paths);
	private:
	};
}
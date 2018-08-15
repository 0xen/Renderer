#pragma once

#include <renderer\IPipeline.hpp>
#include <renderer\ShaderStage.hpp>


namespace Renderer
{
	class IModelPool;
	class IGraphicsPipeline : public virtual IPipeline
	{
	public:
		IGraphicsPipeline(std::map<ShaderStage,const char*> paths);
		virtual void AttachModelPool(IModelPool* model_pool) = 0;
	private:
	};
}
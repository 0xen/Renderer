#pragma once

#include <renderer\IPipeline.hpp>
#include <renderer\ShaderStage.hpp>


namespace Renderer
{
	class IModelPool;
	class VertexBase;
	class IGraphicsPipeline : public virtual IPipeline
	{
	public:
		IGraphicsPipeline(std::map<ShaderStage,const char*> paths);
		virtual void AttachModelPool(IModelPool* model_pool) = 0;
		virtual void AttachVertexBinding(VertexBase vertex_binding) = 0;
	private:
	};
}
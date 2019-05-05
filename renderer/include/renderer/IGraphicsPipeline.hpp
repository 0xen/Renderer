#pragma once

#include <renderer\IPipeline.hpp>
#include <renderer\ShaderStage.hpp>


namespace Renderer
{



	enum PrimitiveTopology
	{
		PointList,
		TriangleList
	};

	class IModelPool;
	class VertexBase;
	class IGraphicsPipeline : public virtual IPipeline
	{
	public:
		IGraphicsPipeline(std::vector<std::pair<Renderer::ShaderStage, const char*>> paths);
		virtual ~IGraphicsPipeline() {};
		virtual void AttachModelPool(IModelPool* model_pool) = 0;
		virtual void AttachVertexBinding(VertexBase vertex_binding) = 0;
		virtual void UseDepth(bool depth) = 0;
		virtual void UseCulling(bool culling) = 0;
		virtual void DefinePrimitiveTopology(PrimitiveTopology top) = 0;
	private:
	};
}
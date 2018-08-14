#pragma once

#include <vector>

namespace Renderer
{
	class IComputePipeline;
	class IComputeProgram
	{
	public:
		IComputeProgram();
		void AttachPipeline(IComputePipeline* pipeline);
		virtual void Build() = 0;
		virtual void Run() = 0;
	protected:
		std::vector<IComputePipeline*> m_pipelines;
	};
}
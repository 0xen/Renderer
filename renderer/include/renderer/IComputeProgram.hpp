#pragma once

#include <vector>

namespace Renderer
{
	class IComputePipeline;
	class IComputeProgram
	{
	public:
		IComputeProgram();
		virtual ~IComputeProgram() {}
		virtual void Build() = 0;
		virtual void Run() = 0;
		void AttachPipeline(IComputePipeline* pipeline);
	protected:
		std::vector<IComputePipeline*> m_pipelines;
	};
}
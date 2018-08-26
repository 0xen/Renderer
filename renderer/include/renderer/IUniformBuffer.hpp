#pragma once

#include <renderer\IBuffer.hpp>
#include <renderer\ShaderStage.hpp>

namespace Renderer
{

	class IUniformBuffer : public virtual IBuffer
	{
	public:
		virtual void GetData() = 0;
		virtual void GetData(unsigned int count) = 0;
		virtual void GetData(unsigned int startIndex, unsigned int count) = 0;
	};
}
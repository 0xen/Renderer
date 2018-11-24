#pragma once

#include <renderer\IBuffer.hpp>
#include <renderer\ShaderStage.hpp>

namespace Renderer
{

	class IUniformBuffer : public virtual IBuffer
	{
	public:
		IUniformBuffer(BufferChain level) : IBuffer(level) {}
		virtual void GetData(BufferSlot slot) = 0;
		virtual void GetData(BufferSlot slot, unsigned int count) = 0;
		virtual void GetData(BufferSlot slot, unsigned int startIndex, unsigned int count) = 0;
	};
}
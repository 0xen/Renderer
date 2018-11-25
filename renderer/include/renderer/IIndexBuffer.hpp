#pragma once

#include <renderer\IBuffer.hpp>

namespace Renderer
{

	class IIndexBuffer : public virtual IBuffer
	{
	public:
		IIndexBuffer(BufferChain level) : IBuffer(level) {}
	};
}
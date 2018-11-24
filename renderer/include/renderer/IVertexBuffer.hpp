#pragma once

#include <renderer\IBuffer.hpp>

namespace Renderer
{

	class IVertexBuffer : public virtual IBuffer
	{
	public:
		IVertexBuffer(BufferChain level) : IBuffer(level) {}
	};
}
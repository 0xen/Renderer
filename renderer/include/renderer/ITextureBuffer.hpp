#pragma once

#include <renderer\IBuffer.hpp>

#include <vector>

namespace Renderer
{
	class ITextureBuffer : public virtual IBuffer
	{
	public:
		ITextureBuffer(BufferChain level) : IBuffer(level){}
		virtual intptr_t GetTextureID() = 0;


	};
}
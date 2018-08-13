#pragma once

#include <renderer\IBuffer.hpp>
#include <renderer\IBufferDescriptor.hpp>

namespace Renderer
{

	class IUniformBuffer : public virtual IBuffer, public virtual IBufferDescriptor
	{
	public:

	};
}
#pragma once

#include <vector>

namespace Renderer
{
	class IBuffer;
	class IDescriptorSet
	{
	public:
		virtual void AttachBuffer(unsigned int location, IBuffer* buffer) = 0;
		virtual std::vector<IBuffer*> GetBuffers() = 0;

		virtual void UpdateSet() = 0;
	};
}
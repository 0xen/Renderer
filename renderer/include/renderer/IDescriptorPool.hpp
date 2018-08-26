#pragma once

#include <renderer\IDescriptor.hpp>
#include <renderer\IDescriptorSet.hpp>

namespace Renderer
{
	class IDescriptorPool
	{
	public:
		virtual IDescriptorSet * CreateDescriptorSet() = 0;
	};
}
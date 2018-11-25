#pragma once

#include <renderer\IDescriptor.hpp>
#include <renderer\IDescriptorSet.hpp>

namespace Renderer
{
	class IDescriptorPool
	{
	public:
		virtual ~IDescriptorPool() {}
		virtual IDescriptorSet * CreateDescriptorSet() = 0;
	};
}
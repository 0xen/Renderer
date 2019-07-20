#pragma once

#include <renderer\vulkan\VulkanHeader.hpp>

namespace Renderer
{
	class VertexBinding
	{
	public:
		VertexBinding(unsigned int location, VkFormat format, unsigned int offset)
		{
			m_location = location;
			m_format = format;
			m_offset = offset;
		}
		unsigned int GetLocation()
		{
			return m_location;
		}
		unsigned int GetOffset()
		{
			return m_offset;
		}
		VkFormat GetFormat()
		{
			return m_format;
		}
	private:
		unsigned int m_location;
		unsigned int m_offset;
		VkFormat m_format;
	};
}
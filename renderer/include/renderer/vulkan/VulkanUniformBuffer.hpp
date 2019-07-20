#pragma once

#include <renderer/vulkan/VulkanBuffer.hpp>

namespace Renderer
{

	namespace Vulkan
	{
		class VulkanUniformBuffer : public VulkanBuffer
		{
		public:

			VulkanUniformBuffer(VulkanDevice* device, BufferChain level, void* dataPtr, unsigned int indexSize, unsigned int elementCount,bool modifiable);
			virtual ~VulkanUniformBuffer();

			virtual void GetData(BufferSlot slot);
			virtual void GetData(BufferSlot slot,unsigned int count);
			virtual void GetData(BufferSlot slot, unsigned int startIndex, unsigned int count);
		};
	}
}
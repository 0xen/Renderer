#pragma once

#include <renderer/vulkan/VulkanBuffer.hpp>
#include <renderer/IUniformBuffer.hpp>

namespace Renderer
{

	namespace Vulkan
	{
		class VulkanUniformBuffer : public IUniformBuffer, public VulkanBuffer
		{
		public:
			VulkanUniformBuffer(VulkanDevice* device, void* dataPtr, unsigned int indexSize, unsigned int elementCount,bool modifiable);
			~VulkanUniformBuffer();

			virtual void GetData();
			virtual void GetData(unsigned int count);
			virtual void GetData(unsigned int startIndex, unsigned int count);
		};
	}
}
#pragma once

#include <renderer/vulkan/VulkanBuffer.hpp>
#include <renderer/vulkan/VulkanBufferDescriptor.hpp>
#include <renderer/IIndexBuffer.hpp>

namespace Renderer
{

	namespace Vulkan
	{
		class VulkanIndexBuffer : public IIndexBuffer, public VulkanBuffer
		{
		public:
			VulkanIndexBuffer(VulkanDevice* device, void* dataPtr, unsigned int indexSize, unsigned int elementCount);
			~VulkanIndexBuffer();

			virtual void SetData();
			virtual void SetData(unsigned int count);
			virtual void SetData(unsigned int startIndex, unsigned int count);

		private:
			void CreateStageingBuffer();
			void DestroyStagingBuffer();
			VulkanBuffer * m_staging_buffer;
		};
	}
}
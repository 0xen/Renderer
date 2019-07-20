#pragma once

#include <renderer/vulkan/VulkanBuffer.hpp>

namespace Renderer
{

	namespace Vulkan
	{
		class VulkanIndexBuffer : public VulkanBuffer
		{
		public:
			VulkanIndexBuffer(VulkanDevice* device, void* dataPtr, unsigned int indexSize, unsigned int elementCount);
			virtual ~VulkanIndexBuffer();

			void SetData(BufferSlot slot);
			void SetData(BufferSlot slot, unsigned int count);
			void SetData(BufferSlot slot, unsigned int startIndex, unsigned int count);

		private:
			void CreateStageingBuffer(BufferSlot slot);
			void DestroyStagingBuffer(BufferSlot slot);
			VulkanBuffer * m_staging_buffer;
			static const BufferChain m_level;
		};
	}
}
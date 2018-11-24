#pragma once

#include <renderer/vulkan/VulkanBuffer.hpp>
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

			virtual void SetData(BufferSlot slot);
			virtual void SetData(BufferSlot slot,unsigned int count);
			virtual void SetData(BufferSlot slot,unsigned int startIndex, unsigned int count);

		private:
			void CreateStageingBuffer(BufferSlot slot);
			void DestroyStagingBuffer(BufferSlot slot);
			VulkanBuffer * m_staging_buffer;
			static const BufferChain m_level;
		};
	}
}
#pragma once

#include <renderer/vulkan/VulkanBuffer.hpp>
#include <renderer/IVertexBuffer.hpp>

namespace Renderer
{

	namespace Vulkan
	{
		class VulkanVertexBuffer : public IVertexBuffer, public VulkanBuffer
		{
		public:
			VulkanVertexBuffer(VulkanDevice* device, void* dataPtr, unsigned int indexSize, unsigned int elementCount);
			virtual ~VulkanVertexBuffer();

			virtual void SetData(BufferSlot slot);
			virtual void SetData(BufferSlot slot, unsigned int count);
			virtual void SetData(BufferSlot slot, unsigned int startIndex, unsigned int count);

		private:
			void CreateStageingBuffer(BufferSlot slot);
			void DestroyStagingBuffer(BufferSlot slot);
			VulkanBuffer * m_staging_buffer;
			static const BufferChain m_level;
		};
	}
}
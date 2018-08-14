#pragma once

#include <renderer/vulkan/VulkanBuffer.hpp>
#include <renderer/vulkan/VulkanBufferDescriptor.hpp>
#include <renderer/IVertexBuffer.hpp>

namespace Renderer
{

	namespace Vulkan
	{
		class VulkanVertexBuffer : public IVertexBuffer, public VulkanBuffer
		{
		public:
			VulkanVertexBuffer(VulkanDevice* device, void* dataPtr, unsigned int indexSize, unsigned int elementCount);
			~VulkanVertexBuffer();

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
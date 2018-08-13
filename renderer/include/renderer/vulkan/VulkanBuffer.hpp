#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>
#include <renderer/vulkan/VulkanBufferData.hpp>
#include <renderer\IBuffer.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanDevice;
		class VulkanBuffer : public virtual IBuffer
		{
		public:
			VulkanBuffer(VulkanDevice* device, void* dataPtr, unsigned int indexSize, unsigned int elementCount, VkBufferUsageFlags _usage, VkMemoryPropertyFlags _memory_propertys_flag);
			~VulkanBuffer();

			virtual void SetData();
			virtual void SetData(unsigned int count);
			virtual void SetData(unsigned int startIndex, unsigned int count);

			virtual void Resize(unsigned int elementCount);

			virtual void GetData();
			virtual void GetData(unsigned int count);
			virtual void GetData(unsigned int startIndex, unsigned int count);
		protected:
			VulkanBufferData m_buffer;
		private:
			VulkanDevice * m_device;
		};
	}
}
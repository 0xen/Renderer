#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>
#include <renderer/vulkan/VulkanBuffer.hpp>
#include <renderer/ITextureBuffer.hpp>
#include <renderer/DataFormat.hpp>

namespace Renderer
{

	namespace Vulkan
	{
		class VulkanTextureBuffer : public ITextureBuffer, public VulkanBuffer, public VulkanStatus
		{
		public:
			VulkanTextureBuffer(VulkanDevice* device, void* dataPtr, DataFormat format, unsigned int width, unsigned int height);
			virtual ~VulkanTextureBuffer();
			VkImage& GetImage();
			virtual void SetData(BufferSlot slot);
			virtual intptr_t GetTextureID();
		private:
			void InitTexture();
			void MoveDataToImage();

			static unsigned int GetFormatSize(DataFormat format);
			static VkFormat GetFormat(DataFormat format);

			DataFormat m_format;
			unsigned int m_width;
			unsigned int m_height;
			int m_mipLevels;

			VkImage m_image;
			VkSampler m_sampler;
			VkImageView m_view;
			VkImageLayout m_image_layout;
			VkDeviceMemory m_device_memory;
			std::vector<VkBufferImageCopy> m_bufferCopyRegions;
		};
	}
}
#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>
#include <renderer/vulkan/VulkanBuffer.hpp>

namespace Renderer
{

	namespace Vulkan
	{
		class VulkanTextureBuffer : public VulkanBuffer, public VulkanStatus
		{
		public:
			VulkanTextureBuffer(VulkanDevice* device, BufferChain level, VkFormat format, unsigned int width, unsigned int height, VkImageUsageFlags usageFlags, VkImageLayout layout);
			VulkanTextureBuffer(VulkanDevice* device, BufferChain level, void* dataPtr, VkFormat format, unsigned int width, unsigned int height, VkImageUsageFlags usageFlags, VkImageLayout layout);
			virtual ~VulkanTextureBuffer();
			VkImage& GetImage();
			virtual void SetData(BufferSlot slot);
			virtual intptr_t GetTextureID();
			unsigned int GetWidth();
			unsigned int GetHeight();
			VkDescriptorImageInfo& GetDescriptorImageInfo(BufferSlot slot);
		private:
			void InitTexture();
			void MoveDataToImage();

			static unsigned int GetFormatSize(VkFormat format);

			VkFormat m_format;
			unsigned int m_width;
			unsigned int m_height;
			int m_mipLevels;

			VkImageUsageFlags m_imageUsageFlags;

			VkImage m_image;
			VkSampler m_sampler;
			VkImageView m_view;
			VkImageLayout m_image_layout;
			VkDeviceMemory m_device_memory;
			std::vector<VkBufferImageCopy> m_bufferCopyRegions;
		};
	}
}
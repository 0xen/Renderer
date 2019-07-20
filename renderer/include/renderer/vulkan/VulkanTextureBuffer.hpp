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
			VulkanTextureBuffer(VulkanDevice* device, void* dataPtr, VkFormat format, unsigned int width, unsigned int height);
			VulkanTextureBuffer(VulkanDevice* device, BufferChain level, void* dataPtr, VkFormat format, unsigned int width, unsigned int height);
			virtual ~VulkanTextureBuffer();
			VkImage& GetImage();
			virtual void SetData(BufferSlot slot);
			virtual intptr_t GetTextureID();
		private:
			void InitTexture();
			void MoveDataToImage();

			static unsigned int GetFormatSize(VkFormat format);

			VkFormat m_format;
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
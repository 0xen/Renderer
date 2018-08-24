#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>
#include <renderer/vulkan/VulkanBuffer.hpp>
#include <renderer/vulkan/VulkanBufferDescriptor.hpp>
#include <renderer/ITextureBuffer.hpp>
#include <renderer/DataFormat.hpp>

namespace Renderer
{

	namespace Vulkan
	{
		class VulkanTextureBuffer : public ITextureBuffer, public VulkanBuffer, public VulkanBufferDescriptor, public VulkanStatus
		{
		public:
			VulkanTextureBuffer(VulkanDevice* device, void* dataPtr, DataFormat format, unsigned int width, unsigned int height, unsigned int binding);
			~VulkanTextureBuffer();
		private:
			void InitTexture();
			static unsigned int GetFormatSize(DataFormat format);
			static VkFormat GetFormat(DataFormat format);

			DataFormat m_format;
			unsigned int m_width;
			unsigned int m_height;

			VkImage m_image;
			VkSampler m_sampler;
			VkImageView m_view;
			VkImageLayout m_image_layout;
			VkDeviceMemory m_device_memory;
		};
	}
}
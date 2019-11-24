#include <renderer/vulkan/VulkanTextureBuffer.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>
#include <renderer/vulkan/VulkanPhysicalDevice.hpp>

using namespace Renderer;
using namespace Renderer::Vulkan;


Renderer::Vulkan::VulkanTextureBuffer::VulkanTextureBuffer(VulkanDevice * device, BufferChain level, VkFormat format, unsigned int width, unsigned int height, VkImageUsageFlags usageFlags, VkImageLayout layout) :
	VulkanBuffer(device, level, nullptr, GetFormatSize(format) * width * height, 1,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
{
	m_imageUsageFlags = usageFlags;
	m_format = format;
	m_width = width;
	m_height = height;

	// Change texture image layout to shader read after all mip levels have been copied
	m_image_layout = layout;

	InitTexture();
	m_gpu_allocation[BufferChain::Single].image_info = VulkanInitializers::DescriptorImageInfo(m_sampler, m_view, m_image_layout);
}

Renderer::Vulkan::VulkanTextureBuffer::VulkanTextureBuffer(VulkanDevice * device, BufferChain level, void * dataPtr, VkFormat format, unsigned int width, unsigned int height, VkImageUsageFlags usageFlags, VkImageLayout layout) :
	VulkanBuffer(device, level, dataPtr, GetFormatSize(format) * width * height, 1,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
{
	m_imageUsageFlags = usageFlags;
	m_format = format;
	m_width = width;
	m_height = height;

	// Change texture image layout to shader read after all mip levels have been copied
	m_image_layout = layout;

	InitTexture();
	VkDeviceSize offset = 0;
	for (unsigned int slot = 0; slot <= (unsigned int)level; slot++)
	{
		m_gpu_allocation[slot].image_info = VulkanInitializers::DescriptorImageInfo(m_sampler, m_view, m_image_layout);
	}
}

Renderer::Vulkan::VulkanTextureBuffer::~VulkanTextureBuffer()
{
	vkDestroyImage(
		*m_device->GetVulkanDevice(),
		m_image,
		nullptr
	);
	vkDestroySampler(
		*m_device->GetVulkanDevice(),
		m_sampler,
		nullptr
	);
	vkDestroyImageView(
		*m_device->GetVulkanDevice(),
		m_view,
		nullptr
	);
	vkFreeMemory(
		*m_device->GetVulkanDevice(),
		m_device_memory,
		nullptr
	);
}

VkImage & Renderer::Vulkan::VulkanTextureBuffer::GetImage()
{
	return m_image;
}

void Renderer::Vulkan::VulkanTextureBuffer::SetData(BufferSlot slot)
{
	if (m_local_allocation[(unsigned int)slot].dataPtr == nullptr) return;
	VulkanBuffer::SetData(slot);
	MoveDataToImage();
}

intptr_t Renderer::Vulkan::VulkanTextureBuffer::GetTextureID()
{
	return (intptr_t)m_image;
}

unsigned int Renderer::Vulkan::VulkanTextureBuffer::GetWidth()
{
	return m_width;
}

unsigned int Renderer::Vulkan::VulkanTextureBuffer::GetHeight()
{
	return m_height;
}

void Renderer::Vulkan::VulkanTextureBuffer::InitTexture()
{

	VkFormatProperties format_properties = m_device->GetVulkanPhysicalDevice()->GetFormatProperties(m_format);
	
	uint32_t offset = 0;

	m_mipLevels = 1;

	// Only dealing with one mip level for now
	VkBufferImageCopy bufferCopyRegion = {};
	bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	bufferCopyRegion.imageSubresource.mipLevel = 0;
	bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
	bufferCopyRegion.imageSubresource.layerCount = 1;
	bufferCopyRegion.imageExtent.width = static_cast<uint32_t>(m_width);
	bufferCopyRegion.imageExtent.height = static_cast<uint32_t>(m_height);
	bufferCopyRegion.imageExtent.depth = 1;
	bufferCopyRegion.bufferOffset = offset;
	m_bufferCopyRegions.push_back(bufferCopyRegion);


	VkImageCreateInfo image_create_info = VulkanInitializers::ImageCreateInfo(
		m_width,
		m_height,
		m_format,
		VK_IMAGE_TILING_OPTIMAL,
		m_imageUsageFlags,
		m_mipLevels
	);

	image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	ErrorCheck(vkCreateImage(
		*m_device->GetVulkanDevice(),
		&image_create_info,
		nullptr,
		&m_image
	));

	VkMemoryRequirements mem_reqs = {};

	vkGetImageMemoryRequirements(
		*m_device->GetVulkanDevice(),
		m_image,
		&mem_reqs
	);


	VkMemoryAllocateInfo mem_alloc_info = VulkanInitializers::MemoryAllocateInfo(mem_reqs.size, VulkanCommon::FindMemoryType(
		m_device->GetVulkanPhysicalDevice(),
		mem_reqs.memoryTypeBits,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
	));

	ErrorCheck(vkAllocateMemory(
		*m_device->GetVulkanDevice(),
		&mem_alloc_info,
		nullptr,
		&m_device_memory
	));

	ErrorCheck(vkBindImageMemory(
		*m_device->GetVulkanDevice(),
		m_image,
		m_device_memory,
		0
	));

	SetData(BufferSlot::Primary);




	VkSamplerCreateInfo sampler_info = VulkanInitializers::SamplerCreateInfo();


	sampler_info.magFilter = VK_FILTER_LINEAR;
	sampler_info.minFilter = VK_FILTER_LINEAR;
	sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_info.mipLodBias = 0.0f;
	sampler_info.compareOp = VK_COMPARE_OP_NEVER;
	sampler_info.minLod = 0.0f;
	// Set max level-of-detail to mip level count of the texture
	sampler_info.maxLod = (float)m_mipLevels;
	// Enable anisotropic filtering
	// This feature is optional, so we must check if it's supported on the device
	if (m_device->GetVulkanPhysicalDevice()->GetDeviceFeatures()->samplerAnisotropy)
	{
		// Use max. level of anisotropy for this example
		sampler_info.maxAnisotropy = m_device->GetVulkanPhysicalDevice()->GetPhysicalDeviceProperties()->limits.maxSamplerAnisotropy;
		sampler_info.anisotropyEnable = VK_TRUE;
	}
	else
	{
		// The device does not support anisotropic filtering
		sampler_info.maxAnisotropy = 1.0;
		sampler_info.anisotropyEnable = VK_FALSE;
	}
	sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	ErrorCheck(vkCreateSampler(
		*m_device->GetVulkanDevice(),
		&sampler_info,
		nullptr,
		&m_sampler
	));

	VkImageViewCreateInfo view_info = VulkanInitializers::ImageViewCreate(m_image, m_format, VK_IMAGE_ASPECT_COLOR_BIT);

	view_info.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };


	// Linear tiling usually won't support mip maps
	// Only set mip map count if optimal tiling is used
	view_info.subresourceRange.levelCount = m_mipLevels;

	ErrorCheck(vkCreateImageView(
		*m_device->GetVulkanDevice(),
		&view_info,
		nullptr,
		&m_view
	));





}

void Renderer::Vulkan::VulkanTextureBuffer::MoveDataToImage()
{
	VkCommandBuffer copy_cmd;
	m_device->GetGraphicsCommand(&copy_cmd, true);

	// The sub resource range describes the regions of the image we will be transition
	VkImageSubresourceRange subresourceRange = {};
	// Image only contains color data
	subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	// Start at first mip level
	subresourceRange.baseMipLevel = 0;
	// We will transition on all mip levels
	subresourceRange.levelCount = m_mipLevels;
	// The 2D texture only has one layer
	subresourceRange.layerCount = 1;

	// Optimal image will be used as destination for the copy, so we must transfer from our
	// initial undefined image layout to the transfer destination layout

	VulkanCommon::SetImageLayout(
		copy_cmd,
		m_image,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		subresourceRange//,
		//VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		//VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
	);


	// Copy mip levels from staging buffer
	vkCmdCopyBufferToImage(
		copy_cmd,
		GetBufferData(BufferSlot::Primary)->buffer,
		m_image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		static_cast<uint32_t>(m_bufferCopyRegions.size()),
		m_bufferCopyRegions.data());


	VulkanCommon::SetImageLayout(
		copy_cmd,
		m_image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		m_image_layout,
		subresourceRange//,
		//VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		//VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
		);


	ErrorCheck(vkEndCommandBuffer(copy_cmd));


	m_device->SubmitGraphicsCommand(&copy_cmd, 1);

	m_device->FreeGraphicsCommand(&copy_cmd, 1);
}

unsigned int Renderer::Vulkan::VulkanTextureBuffer::GetFormatSize(VkFormat format)
{
	
	switch (format)
	{
		case VkFormat::VK_FORMAT_R8G8_UNORM:
		case VkFormat::VK_FORMAT_R8G8_SNORM:
		case VkFormat::VK_FORMAT_R8G8_USCALED:
		case VkFormat::VK_FORMAT_R8G8_SSCALED:
		case VkFormat::VK_FORMAT_R8G8_UINT:
		case VkFormat::VK_FORMAT_R8G8_SINT:
		case VkFormat::VK_FORMAT_R8G8_SRGB:
			return 2;
			break;
		case VkFormat::VK_FORMAT_R8G8B8_UNORM:
		case VkFormat::VK_FORMAT_R8G8B8_SNORM:
		case VkFormat::VK_FORMAT_R8G8B8_USCALED:
		case VkFormat::VK_FORMAT_R8G8B8_SSCALED:
		case VkFormat::VK_FORMAT_R8G8B8_UINT:
		case VkFormat::VK_FORMAT_R8G8B8_SINT:
		case VkFormat::VK_FORMAT_R8G8B8_SRGB:
			return 3;
			break;
		case VkFormat::VK_FORMAT_R8G8B8A8_UNORM:
		case VkFormat::VK_FORMAT_R8G8B8A8_SNORM:
		case VkFormat::VK_FORMAT_R8G8B8A8_USCALED:
		case VkFormat::VK_FORMAT_R8G8B8A8_SSCALED:
		case VkFormat::VK_FORMAT_R8G8B8A8_UINT:
		case VkFormat::VK_FORMAT_R8G8B8A8_SINT:
		case VkFormat::VK_FORMAT_R8G8B8A8_SRGB:
			return 4;
			break;
	}
	return 0;
}

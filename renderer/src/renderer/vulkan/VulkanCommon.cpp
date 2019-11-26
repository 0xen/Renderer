#include <renderer/vulkan/VulkanCommon.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanPhysicalDevice.hpp>
#include <renderer/vulkan/VulkanBufferData.hpp>

#include <fstream>

using namespace Renderer::Vulkan;

void VulkanCommon::CreateImageView(VulkanDevice* device, VkImage image, VkFormat format, VkImageAspectFlags aspect_flags, VkImageView & view)
{
	VkImageViewCreateInfo create_info = VulkanInitializers::ImageViewCreate(image, format, aspect_flags);
	vkCreateImageView(
		*device->GetVulkanDevice(),
		&create_info,
		nullptr,
		&view
	);
}

VkFormat VulkanCommon::GetDepthImageFormat(VulkanDevice * device)
{
	return FindSupportedFormat(
		device,
		{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },	// Formats
		VK_IMAGE_TILING_OPTIMAL,															// Tiling
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT										// Features
	);
}

VkFormat VulkanCommon::FindSupportedFormat(VulkanDevice * device, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
	for (VkFormat format : candidates)
	{
		VkFormatProperties props;
		vkGetPhysicalDeviceFormatProperties(*device->GetVulkanPhysicalDevice()->GetPhysicalDevice(), format, &props);
		if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
		{
			return format;
		}
		else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
		{
			return format;
		}
	}
	assert(0 && "All formats are not supported");
	return VK_FORMAT_UNDEFINED;
}

void VulkanCommon::CreateImage(VulkanDevice* device, VkExtent2D extent, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage & image, VkDeviceMemory & image_memory)
{
	VkImageCreateInfo create_info = VulkanInitializers::ImageCreateInfo(extent.width, extent.height, format, tiling, usage);
	vkCreateImage(
		*device->GetVulkanDevice(),
		&create_info,
		nullptr,
		&image
	);

	VkMemoryRequirements mem_requirements;
	vkGetImageMemoryRequirements(
		*device->GetVulkanDevice(),
		image,
		&mem_requirements
	);

	VkMemoryAllocateInfo alloc_info = VulkanInitializers::MemoryAllocateInfo(mem_requirements.size, FindMemoryType(
		device->GetVulkanPhysicalDevice(),
		mem_requirements.memoryTypeBits,
		properties
	));

	vkAllocateMemory(
		*device->GetVulkanDevice(),
		&alloc_info,
		nullptr,
		&image_memory
	);

	vkBindImageMemory(
		*device->GetVulkanDevice(),
		image,
		image_memory,
		0
	);
}

uint32_t VulkanCommon::FindMemoryType(VulkanPhysicalDevice * device, uint32_t type_filter, VkMemoryPropertyFlags properties)
{
	for (uint32_t i = 0; i < device->GetPhysicalDeviceMemoryProperties()->memoryTypeCount; i++)
	{
		if (type_filter & (1 << i) &&
			(device->GetPhysicalDeviceMemoryProperties()->memoryTypes[i].propertyFlags & properties) == properties)
		{
			return i;
		}
	}
	assert(0 && "No available memory properties");
	return -1;
}

void Renderer::Vulkan::VulkanCommon::TransitionImageLayout(VkCommandBuffer cmd, VkImage image, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout,
	VkPipelineStageFlags srcStageMask,
	VkPipelineStageFlags dstStageMask, VkImageSubresourceRange subresourceRange)
{
	VkImageMemoryBarrier barrier = VulkanInitializers::ImageMemoryBarrier(image, format, old_layout, new_layout);

	barrier.subresourceRange = subresourceRange;

	vkCmdPipelineBarrier(
		cmd,
		srcStageMask,
		dstStageMask,
		0,
		0,
		nullptr,
		0,
		nullptr,
		1,
		&barrier
	);
}

void VulkanCommon::TransitionImageLayout(VulkanDevice* device, VkImage image, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout, VkImageSubresourceRange subresourceRange)
{
	VkCommandBuffer command_buffer = BeginSingleTimeCommands(device,*device->GetGraphicsCommandPool());

	VkImageMemoryBarrier barrier = VulkanInitializers::ImageMemoryBarrier(image, format, old_layout, new_layout);

	barrier.subresourceRange = subresourceRange;

	vkCmdPipelineBarrier(
		command_buffer,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		0,
		0,
		nullptr,
		0,
		nullptr,
		1,
		&barrier
	);
	EndSingleTimeCommands(device,command_buffer, *device->GetGraphicsCommandPool());
}

VkCommandBuffer VulkanCommon::BeginSingleTimeCommands(VulkanDevice * device, VkCommandPool command_pool)
{
	VkCommandBufferAllocateInfo alloc_info = VulkanInitializers::CommandBufferAllocateInfo(
		command_pool,
		1
	);
	VkCommandBuffer command_buffer;
	vkAllocateCommandBuffers(
		*device->GetVulkanDevice(),
		&alloc_info,
		&command_buffer
	);
	VkCommandBufferBeginInfo begin_info = VulkanInitializers::CommandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	vkBeginCommandBuffer(
		command_buffer,
		&begin_info
	);
	return command_buffer;
}

VkCommandBuffer Renderer::Vulkan::VulkanCommon::BeginSimultaneousCommand(VulkanDevice * device, VkCommandPool command_pool)
{
	VkCommandBufferAllocateInfo alloc_info = VulkanInitializers::CommandBufferAllocateInfo(
		command_pool,
		1
	);
	VkCommandBuffer command_buffer;
	vkAllocateCommandBuffers(
		*device->GetVulkanDevice(),
		&alloc_info,
		&command_buffer
	);
	VkCommandBufferBeginInfo begin_info = VulkanInitializers::CommandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
	vkBeginCommandBuffer(
		command_buffer,
		&begin_info
	);
	return command_buffer;
}

void VulkanCommon::EndSingleTimeCommands(VulkanDevice * device, VkCommandBuffer command_buffer, VkCommandPool command_pool)
{
	vkEndCommandBuffer(command_buffer);
	VkSubmitInfo submit_info = VulkanInitializers::SubmitInfo(command_buffer);
	VkResult res = vkQueueSubmit(
		*device->GetGraphicsQueue(),
		1,
		&submit_info,
		VK_NULL_HANDLE
	);
	vkQueueWaitIdle(
		*device->GetGraphicsQueue()
	);
	vkFreeCommandBuffers(
		*device->GetVulkanDevice(),
		command_pool,
		1,
		&command_buffer
	);
}

bool VulkanCommon::HasStencilComponent(VkFormat format)
{
	return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

void Renderer::Vulkan::VulkanCommon::CreateBuffer(VulkanDevice * device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VulkanBufferData & buffer)
{
	VkBufferCreateInfo buffer_info = VulkanInitializers::BufferCreateInfo(size, usage);

	VkResult a = vkCreateBuffer(
		*device->GetVulkanDevice(),
		&buffer_info,
		nullptr,
		&buffer.buffer
	);

	VkMemoryRequirements mem_requirements;
	vkGetBufferMemoryRequirements(
		*device->GetVulkanDevice(),
		buffer.buffer,
		&mem_requirements
	);
	

	VkMemoryAllocateInfo alloc_info = VulkanInitializers::MemoryAllocateInfo(mem_requirements.size, FindMemoryType(
		device->GetVulkanPhysicalDevice(),
		mem_requirements.memoryTypeBits,
		properties
	));
	buffer.size = mem_requirements.size > size ? size : mem_requirements.size;
	buffer.alignment = mem_requirements.alignment;
	a = vkAllocateMemory(
		*device->GetVulkanDevice(),
		&alloc_info,
		nullptr,
		&buffer.buffer_memory
	);

	vkBindBufferMemory(
		*device->GetVulkanDevice(),
		buffer.buffer,
		buffer.buffer_memory,
		0
	);


}

void Renderer::Vulkan::VulkanCommon::MapBufferMemory(VulkanDevice* device, VulkanBufferData & buffer, VkDeviceSize size)
{
	vkMapMemory(*device->GetVulkanDevice(), buffer.buffer_memory, 0, size, 0, &buffer.mapped_memory);
}

void Renderer::Vulkan::VulkanCommon::UnMapBufferMemory(VulkanDevice * device, VulkanBufferData & buffer)
{
	vkUnmapMemory(*device->GetVulkanDevice(), buffer.buffer_memory);
	buffer.mapped_memory = nullptr;
}

void Renderer::Vulkan::VulkanCommon::DestroyBuffer(VulkanDevice * device, VulkanBufferData & buffer)
{

	vkDestroyBuffer(
		*device->GetVulkanDevice(),
		buffer.buffer,
		nullptr
	);
	vkFreeMemory(
		*device->GetVulkanDevice(),
		buffer.buffer_memory,
		nullptr
	);
}

std::vector<char> Renderer::Vulkan::VulkanCommon::ReadFile(const std::string & filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open())
	{
		throw std::runtime_error("failed to open file!");
	}
	size_t file_size = (size_t)file.tellg();
	std::vector<char> buffer(file_size);
	file.seekg(0);
	file.read(buffer.data(), file_size);
	file.close();
	return buffer;
}

VkShaderModule Renderer::Vulkan::VulkanCommon::CreateShaderModule(VulkanDevice * device, const std::vector<char>& code)
{
	VkShaderModuleCreateInfo create_info = VulkanInitializers::ShaderModuleCreateInfo(code);

	VkShaderModule shader_module;
	vkCreateShaderModule(
		*device->GetVulkanDevice(),
		&create_info,
		nullptr,
		&shader_module
	);

	return shader_module;
}

void Renderer::Vulkan::VulkanCommon::CopyBuffer(VulkanDevice * device, VkBuffer from_buffer, VkBuffer to_buffer, VkDeviceSize size)
{
	VkCommandBufferAllocateInfo alloc_info = VulkanInitializers::CommandBufferAllocateInfo(*device->GetGraphicsCommandPool(), 1);

	VkCommandBuffer command_buffer;
	vkAllocateCommandBuffers(
		*device->GetVulkanDevice(),
		&alloc_info,
		&command_buffer
	);

	VkCommandBufferBeginInfo begin_info = VulkanInitializers::CommandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	vkBeginCommandBuffer(
		command_buffer,
		&begin_info
	);

	// Copy the temp Ram buffer to the device memory
	VkBufferCopy copy_region = {};
	copy_region.srcOffset = 0; // Optional
	copy_region.dstOffset = 0; // Optional
	copy_region.size = size;
	vkCmdCopyBuffer(
		command_buffer,
		from_buffer,
		to_buffer,
		1,
		&copy_region
	);
	vkEndCommandBuffer(
		command_buffer
	);
	// Now the command has been recorded, submit it

	VkSubmitInfo submit_info = VulkanInitializers::SubmitInfo(command_buffer);

	vkQueueSubmit(
		*device->GetGraphicsQueue(),
		1,
		&submit_info,
		VK_NULL_HANDLE
	);
	vkQueueWaitIdle(
		*device->GetGraphicsQueue()
	);

	vkFreeCommandBuffers(
		*device->GetVulkanDevice(),
		*device->GetGraphicsCommandPool(),
		1,
		&command_buffer
	);
}

void Renderer::Vulkan::VulkanCommon::SetImageLayout(VkCommandBuffer cmdbuffer, VkImage image, VkImageLayout oldImageLayout, VkImageLayout newImageLayout, VkImageSubresourceRange subresourceRange)
{
	VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	// Create an image barrier object
	VkImageMemoryBarrier imageMemoryBarrier = VulkanInitializers::ImageMemoryBarrier();
	imageMemoryBarrier.oldLayout = oldImageLayout;
	imageMemoryBarrier.newLayout = newImageLayout;
	imageMemoryBarrier.image = image;
	imageMemoryBarrier.subresourceRange = subresourceRange;

	// Source layouts (old)
	// Source access mask controls actions that have to be finished on the old layout
	// before it will be transitioned to the new layout
	/*switch (oldImageLayout)
	{
	case VK_IMAGE_LAYOUT_UNDEFINED:
		// Image layout is undefined (or does not matter)
		// Only valid as initial layout
		// No flags required, listed only for completeness
		imageMemoryBarrier.srcAccessMask = 0;
		break;

	case VK_IMAGE_LAYOUT_PREINITIALIZED:
		// Image is preinitialized
		// Only valid as initial layout for linear images, preserves memory contents
		// Make sure host writes have been finished
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
		break;

	case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
		// Image is a color attachment
		// Make sure any writes to the color buffer have been finished
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		break;

	case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
		// Image is a depth/stencil attachment
		// Make sure any writes to the depth/stencil buffer have been finished
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		break;

	case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
		// Image is a transfer source 
		// Make sure any reads from the image have been finished
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		break;

	case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
		// Image is a transfer destination
		// Make sure any writes to the image have been finished
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		break;

	case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
		// Image is read by a shader
		// Make sure any shader reads from the image have been finished
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		break;
	default:
		// Other source layouts aren't handled (yet)
		break;
	}

	// Target layouts (new)
	// Destination access mask controls the dependency for the new image layout
	switch (newImageLayout)
	{
	case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
		// Image will be used as a transfer destination
		// Make sure any writes to the image have been finished
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		break;

	case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
		// Image will be used as a transfer source
		// Make sure any reads from the image have been finished
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		break;

	case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
		// Image will be used as a color attachment
		// Make sure any writes to the color buffer have been finished
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		break;

	case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
		// Image layout will be used as a depth/stencil attachment
		// Make sure any writes to depth/stencil buffer have been finished
		imageMemoryBarrier.dstAccessMask = imageMemoryBarrier.dstAccessMask | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		break;

	case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
		// Image will be read in a shader (sampler, input attachment)
		// Make sure any writes to the image have been finished
		if (imageMemoryBarrier.srcAccessMask == 0)
		{
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
		}
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		break;
	default:
		// Other source layouts aren't handled (yet)
		break;
	}*/


	if(oldImageLayout == VK_IMAGE_LAYOUT_UNDEFINED && newImageLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
	{
		imageMemoryBarrier.srcAccessMask = 0;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (oldImageLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newImageLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	{
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
		dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else
	{
		//throw std::invalid_argument("unsupported layout transition!");
	}





	// Put barrier inside setup command buffer
	vkCmdPipelineBarrier(
		cmdbuffer,
		srcStageMask,
		dstStageMask,
		0,
		0, nullptr,
		0, nullptr,
		1, &imageMemoryBarrier);
}

void Renderer::Vulkan::VulkanCommon::SetImageLayout(VkCommandBuffer cmdbuffer, VkImage image, VkImageAspectFlags aspectMask, VkImageLayout oldImageLayout, VkImageLayout newImageLayout)
{
	VkImageSubresourceRange subresourceRange = {};
	subresourceRange.aspectMask = aspectMask;
	subresourceRange.baseMipLevel = 0;
	subresourceRange.levelCount = 1;
	subresourceRange.layerCount = 1;
	SetImageLayout(cmdbuffer, image, oldImageLayout, newImageLayout, subresourceRange);
}

VkDevice s_global_device = VK_NULL_HANDLE;

VKAPI_ATTR VkResult VKAPI_CALL
vkCreateRayTracingPipelinesNV(VkDevice      device,
	VkPipelineCache                         pipelineCache,
	uint32_t                                createInfoCount,
	const VkRayTracingPipelineCreateInfoNV* pCreateInfos,
	const VkAllocationCallbacks*            pAllocator,
	VkPipeline*                             pPipelines)
{
	s_global_device = device;
	static const auto call = reinterpret_cast<PFN_vkCreateRayTracingPipelinesNV>(
		vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesNV"));
	return call(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
}

VKAPI_ATTR VkResult VKAPI_CALL
vkCreateAccelerationStructureNV(VkDevice       device,
	const VkAccelerationStructureCreateInfoNV* pCreateInfo,
	const VkAllocationCallbacks*               pAllocator,
	VkAccelerationStructureNV*                 pAccelerationStructure)
{
	s_global_device = device;
	static const auto call = reinterpret_cast<PFN_vkCreateAccelerationStructureNV>(
		vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureNV"));
	return call(device, pCreateInfo, pAllocator, pAccelerationStructure);
}

VKAPI_ATTR void VKAPI_CALL vkGetAccelerationStructureMemoryRequirementsNV(
	VkDevice                                               device,
	const VkAccelerationStructureMemoryRequirementsInfoNV* pInfo,
	VkMemoryRequirements2KHR*                              pMemoryRequirements)
{
	s_global_device = device;
	static const auto call = reinterpret_cast<PFN_vkGetAccelerationStructureMemoryRequirementsNV>(
		vkGetDeviceProcAddr(device, "vkGetAccelerationStructureMemoryRequirementsNV"));
	return call(device, pInfo, pMemoryRequirements);
}


VKAPI_ATTR VkResult VKAPI_CALL
vkBindAccelerationStructureMemoryNV(VkDevice       device,
	uint32_t                                       bindInfoCount,
	const VkBindAccelerationStructureMemoryInfoNV* pBindInfos)
{
	s_global_device = device;
	static const auto call = reinterpret_cast<PFN_vkBindAccelerationStructureMemoryNV>(
		vkGetDeviceProcAddr(device, "vkBindAccelerationStructureMemoryNV"));
	return call(device, bindInfoCount, pBindInfos);
}

VKAPI_ATTR void VKAPI_CALL
vkCmdBuildAccelerationStructureNV(
	VkCommandBuffer                      commandBuffer,
	const VkAccelerationStructureInfoNV* pInfo,
	VkBuffer                             instanceData,
	VkDeviceSize                         instanceOffset,
	VkBool32                             update,
	VkAccelerationStructureNV            dst,
	VkAccelerationStructureNV            src,
	VkBuffer                             scratch,
	VkDeviceSize                         scratchOffset)
{
	static const auto call = reinterpret_cast<PFN_vkCmdBuildAccelerationStructureNV>(
		vkGetDeviceProcAddr(s_global_device, "vkCmdBuildAccelerationStructureNV"));
	return call(commandBuffer, pInfo, instanceData, instanceOffset, update, dst, src, scratch,
		scratchOffset);
}

VKAPI_ATTR VkResult VKAPI_CALL
vkGetAccelerationStructureHandleNV(VkDevice device,
	VkAccelerationStructureNV accelerationStructure,
	size_t                    dataSize,
	void*                     pData)
{
	static const auto call = reinterpret_cast<PFN_vkGetAccelerationStructureHandleNV>(
		vkGetDeviceProcAddr(device, "vkGetAccelerationStructureHandleNV"));
	return call(device, accelerationStructure, dataSize, pData);
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetRayTracingShaderGroupHandlesNV(VkDevice   device,
	VkPipeline pipeline,
	uint32_t   firstGroup,
	uint32_t   groupCount,
	size_t     dataSize,
	void*      pData)
{
	static const auto call = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesNV>(
		vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesNV"));
	return call(device, pipeline, firstGroup, groupCount, dataSize, pData);
}

VKAPI_ATTR void VKAPI_CALL vkCmdTraceRaysNV(VkCommandBuffer commandBuffer,
	VkBuffer        raygenShaderBindingTableBuffer,
	VkDeviceSize    raygenShaderBindingOffset,
	VkBuffer        missShaderBindingTableBuffer,
	VkDeviceSize    missShaderBindingOffset,
	VkDeviceSize    missShaderBindingStride,
	VkBuffer        hitShaderBindingTableBuffer,
	VkDeviceSize    hitShaderBindingOffset,
	VkDeviceSize    hitShaderBindingStride,
	VkBuffer        callableShaderBindingTableBuffer,
	VkDeviceSize    callableShaderBindingOffset,
	VkDeviceSize    callableShaderBindingStride,
	uint32_t        width,
	uint32_t        height,
	uint32_t        depth)
{
	static const auto call = reinterpret_cast<PFN_vkCmdTraceRaysNV>(
		vkGetDeviceProcAddr(s_global_device, "vkCmdTraceRaysNV"));
	return call(commandBuffer, raygenShaderBindingTableBuffer, raygenShaderBindingOffset,
		missShaderBindingTableBuffer, missShaderBindingOffset, missShaderBindingStride,
		hitShaderBindingTableBuffer, hitShaderBindingOffset, hitShaderBindingStride,
		callableShaderBindingTableBuffer, callableShaderBindingOffset,
		callableShaderBindingStride, width, height, depth);
}
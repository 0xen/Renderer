#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>

#include <assert.h>
#include <vector>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanDevice;
		class VulkanPhysicalDevice;
		struct VulkanBufferData;
		namespace VulkanCommon
		{

			void CreateImageView(VulkanDevice* device, VkImage image, VkFormat format, VkImageAspectFlags aspect_flags, VkImageView & view);

			VkFormat GetDepthImageFormat(VulkanDevice * device);

			VkFormat FindSupportedFormat(VulkanDevice* device, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

			void CreateImage(VulkanDevice* device, VkExtent2D extent, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage & image, VkDeviceMemory & image_memory);

			void CreateImage(VulkanDevice* device, VkExtent2D extent, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage & image, VkDeviceMemory & image_memory, VkImageLayout initialLayout);

			uint32_t FindMemoryType(VulkanPhysicalDevice* device, uint32_t type_filter, VkMemoryPropertyFlags properties);

			void TransitionImageLayout(VkCommandBuffer cmd, VkImage image, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout,
				VkPipelineStageFlags srcStageMask,
				VkPipelineStageFlags dstStageMask, VkImageSubresourceRange subresourceRange = {});

			void TransitionImageLayout(VulkanDevice* device, VkImage image, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout, VkImageSubresourceRange subresourceRange = {});

			VkCommandBuffer BeginSingleTimeCommands(VulkanDevice* device, VkCommandPool command_pool);

			VkCommandBuffer BeginSimultaneousCommand(VulkanDevice* device, VkCommandPool command_pool);

			void EndSingleTimeCommands(VulkanDevice* device, VkCommandBuffer command_buffer, VkCommandPool command_pool);

			bool HasStencilComponent(VkFormat format);

			void CreateBuffer(VulkanDevice* device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VulkanBufferData & buffer);

			void MapBufferMemory(VulkanDevice* device, VulkanBufferData& buffer, VkDeviceSize size);

			void UnMapBufferMemory(VulkanDevice* device, VulkanBufferData& buffer);

			void DestroyBuffer(VulkanDevice * device, VulkanBufferData & buffer);

			std::vector<char> ReadFile(const std::string & filename);

			VkShaderModule CreateShaderModule(VulkanDevice * device, const std::vector<char>& code);

			void CopyBuffer(VulkanDevice * device, VkBuffer from_buffer, VkBuffer to_buffer, VkDeviceSize size);

			// Sorced from https://github.com/SaschaWillems/Vulkan/blob/master/base/VulkanTools.cpp
			void SetImageLayout(VkCommandBuffer cmdbuffer, VkImage image, VkImageLayout oldImageLayout, VkImageLayout newImageLayout, VkImageSubresourceRange subresourceRange);

			void SetImageLayout(VkCommandBuffer cmdbuffer, VkImage image, VkImageAspectFlags aspectMask, VkImageLayout oldImageLayout, VkImageLayout newImageLayout);



			//VKAPI_ATTR VkResult VKAPI_CALL vkCreateRayTracingPipelinesNV(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkRayTracingPipelineCreateInfoNV* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines);




		}
	}
	
}
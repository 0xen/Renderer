#pragma once

#include <vector>
#include <array>

#include <renderer/NativeWindowHandle.hpp>
#include <renderer/vulkan/VulkanHeader.hpp>
#include <renderer/vulkan/VulkanSwapChainSupport.hpp>
#include <renderer/vulkan/VulkanQueueFamilyIndices.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		namespace VulkanInitializers
		{
			VkApplicationInfo ApplicationInfo(const char* app_name, uint32_t app_ver, const char* engine_name, uint32_t engine_ver, uint32_t api_version);

			VkInstanceCreateInfo InstanceCreateInfo(VkApplicationInfo& app_info, std::vector<const char*>& instance_extensions, std::vector<const char*>& instance_layers);

			VkWin32SurfaceCreateInfoKHR SurfaceCreateInfo(Renderer::NativeWindowHandle* window_handle);

			VkDeviceQueueCreateInfo DeviceQueueCreate(uint32_t queue_family_index, float queue_priority);

			VkDeviceCreateInfo DeviceCreateInfo(std::vector<VkDeviceQueueCreateInfo>& queue_create_infos, const std::vector<const char*>& device_extensions, VkPhysicalDeviceFeatures& device_features);

			VkCommandPoolCreateInfo CommandPoolCreateInfo(uint32_t queue_family_index, VkCommandPoolCreateFlags flags = 0);

			VkCommandBufferAllocateInfo CommandBufferAllocateInfo(VkCommandPool pool, uint32_t command_buffer_count);

			VkCommandBufferBeginInfo CommandBufferBeginInfo(VkCommandBufferUsageFlags flag);

			VkSubmitInfo SubmitInfo(VkCommandBuffer& buffer);

			VkSubmitInfo SubmitInfo(VkCommandBuffer* buffer, uint32_t count = 1);

			VkRenderPassBeginInfo RenderPassBeginInfo(VkRenderPass render_pass, VkExtent2D swapchain_extent, std::array<VkClearValue, 2>& clear_values);

			VkSwapchainCreateInfoKHR SwapchainCreateInfoKHR(VkSurfaceFormatKHR surface_format, VkExtent2D extent, VkPresentModeKHR present_mode, uint32_t image_count, VkSurfaceKHR surface, Renderer::Vulkan::VulkanQueueFamilyIndices indices, Renderer::Vulkan::VulkanSwapChainSupport swap_chain_support);

			VkSubpassDependency SubpassDependency();

			VkRenderPassCreateInfo RenderPassCreateInfo(std::vector<VkAttachmentDescription>& color_attachment, VkSubpassDescription& subpass, VkSubpassDependency& subpass_dependency);

			VkAttachmentDescription AttachmentDescription(VkFormat format, VkAttachmentStoreOp store_op, VkImageLayout final_layout);

			VkAttachmentReference AttachmentReference(VkImageLayout layout, uint32_t attachment);

			VkSubpassDescription SubpassDescription(VkAttachmentReference& color_attachment_refrence, VkAttachmentReference& depth_attachment_ref);

			VkFramebufferCreateInfo FramebufferCreateInfo(VkExtent2D& swap_chain_extent, std::vector<VkImageView>& attachments, VkRenderPass& render_pass);

			VkSemaphoreCreateInfo SemaphoreCreateInfo();

			VkImageViewCreateInfo ImageViewCreate(VkImage image, VkFormat format, VkImageAspectFlags aspect_flags);

			VkMemoryAllocateInfo MemoryAllocateInfo(VkDeviceSize size, uint32_t memory_type_index);

			VkImageCreateInfo ImageCreateInfo(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, uint32_t mip_levels = 1);

			VkImageMemoryBarrier ImageMemoryBarrier();

			VkImageMemoryBarrier ImageMemoryBarrier(VkImage& image, VkFormat& format, VkImageLayout& old_layout, VkImageLayout& new_layout);

			VkBufferCreateInfo BufferCreateInfo(VkDeviceSize size, VkBufferUsageFlags usage);

			VkDescriptorBufferInfo DescriptorBufferInfo(VkBuffer buffer, uint32_t size, VkDeviceSize & offset);

			VkDescriptorPoolSize DescriptorPoolSize(VkDescriptorType type);

			VkDescriptorSetLayoutBinding DescriptorSetLayoutBinding(VkDescriptorType type, VkShaderStageFlags stage_flags, uint32_t binding);

			VkDescriptorPoolCreateInfo DescriptorPoolCreateInfo(std::vector<VkDescriptorPoolSize>& pool_sizes, uint32_t max_sets);

			VkDescriptorSetLayoutCreateInfo DescriptorSetLayoutCreateInfo(std::vector<VkDescriptorSetLayoutBinding>& layout_bindings);

			VkPipelineLayoutCreateInfo PipelineLayoutCreateInfo(VkDescriptorSetLayout & descriptor_set_layout);

			VkDescriptorSetAllocateInfo DescriptorSetAllocateInfo(std::vector<VkDescriptorSetLayout>& layouts, VkDescriptorPool & pool);

			VkShaderModuleCreateInfo ShaderModuleCreateInfo(const std::vector<char>& code);

			VkPipelineShaderStageCreateInfo PipelineShaderStageCreateInfo(VkShaderModule & shader, const char * main, VkShaderStageFlagBits flag);

			VkComputePipelineCreateInfo ComputePipelineCreateInfo(VkPipelineLayout & layout, VkPipelineShaderStageCreateInfo & shader_stage);

			VkWriteDescriptorSet WriteDescriptorSet(VkDescriptorSet d_set, VkDescriptorImageInfo& image_info, VkDescriptorType type, int binding);

			VkWriteDescriptorSet WriteDescriptorSet(VkDescriptorSet d_set, VkDescriptorBufferInfo & buffer_info, VkDescriptorType type, int binding);










		}
	}
}
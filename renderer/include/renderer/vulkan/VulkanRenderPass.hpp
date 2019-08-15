#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>

#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanStatus.hpp>

#include <map>

namespace Renderer
{
	struct NativeWindowHandle;
	namespace Vulkan
	{
		class VulkanDevice;
		class VulkanSwapchain;
		class VulkanDescriptorPool;
		class VulkanDescriptorSet;
		class VulkanRenderer;
		class VulkanInstance;
		class VulkanGraphicsPipeline; 
		class VulkanRenderPass : public VulkanStatus
		{
		public:
			VulkanRenderPass(VulkanRenderer* renderer, VulkanSwapchain* swapchain,  VulkanInstance* instance, VulkanDevice* device, unsigned int subpass_count = 1);
			~VulkanRenderPass();

			// When a change has been made to the current pipeline pool, RebuildCommandBuffers() must be called
			void AttachGraphicsPipeline(VulkanGraphicsPipeline* pipeline);
			// When a change has been made to the current pipeline pool, RebuildCommandBuffers() must be called
			void RemoveGraphicsPipeline(VulkanGraphicsPipeline* pipeline);

			void RebuildCommandBuffers();

			void RequestRebuildCommandBuffers();

			void Render();

			void FindNextImageIndex();

			unsigned int GetCurrentFrameIndex();

			VulkanDescriptorPool* GetInputAttachmentsReadPool();
			
			void SubmitQueue(unsigned int currentBuffer);

			VkRenderPass& GetRenderPass();
		private:


			void InitRenderPass();
			void DeInitRenderPass();

			// Frame buffer
			void InitFrameBuffer();
			void DeInitFrameBuffer();

			// Command buffers
			void InitCommandBuffers();

			// Create render fences
			void InitFences();

			// Structure for attachments was provided by https://github.com/SaschaWillems/Vulkan/blob/master/examples/inputattachments/inputattachments.cpp
			struct FrameBufferAttachment
			{
				VkImage image;
				VkDeviceMemory memory;
				VkImageView view;
				VkFormat format;
			};

			// Create the attachments for each swapchain image
			void CreateAttachmentImages(VkFormat format, VkImageUsageFlags usage, FrameBufferAttachment& attachment);

			VulkanInstance* m_instance;
			VulkanRenderer* m_renderer;
			VulkanDevice* m_device;
			VulkanSwapchain* m_swapchain;


			// Attachments
			struct Attachments
			{
				FrameBufferAttachment color, depth;
			};
			std::vector<Attachments> m_attachments;

			VulkanDescriptorPool* m_input_attachments_read_pool;
			// Define the various attachments that will be passed throughout the sub passes
			std::vector<VulkanDescriptorSet*> m_input_attachments_read_sets;


			// Render pass
			VkRenderPass m_render_pass;
			unsigned int m_subpass_count;


			// Frame buffers
			std::vector<VkFramebuffer> m_swap_chain_framebuffers;

			 
			// Command buffers
			std::vector<VkCommandBuffer> m_command_buffers;

			// What render pass are they in and then what pipelines to call in that render pass
			std::map<unsigned int, std::vector<VulkanGraphicsPipeline*>> m_subpasses;

			// Render fence
			std::vector<VkFence> m_fence;

			// Current frame index we are rendering too
			uint32_t m_frame_index = 0;

			bool m_should_rebuild_cmd;
		};
	}
}
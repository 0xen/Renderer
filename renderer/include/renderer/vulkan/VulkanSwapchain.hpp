#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>

#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanStatus.hpp>
#include <renderer/vulkan/VulkanSwapChainSupport.hpp>

namespace Renderer
{
	struct NativeWindowHandle;
	namespace Vulkan
	{
		class VulkanInstance;
		class VulkanDevice;
		class VulkanGraphicsPipeline;
		class VulkanSwapchain : public VulkanStatus
		{
		public:
			VulkanSwapchain(VulkanInstance* instance, VulkanDevice* device, VkSurfaceKHR* surface, Renderer::NativeWindowHandle* window_handle);
			~VulkanSwapchain();
			void RequestRebuildCommandBuffers();
			void RebuildSwapchain();
			unsigned int GetCurrentFrameIndex();
			VkImageView GetBackBufferImage(unsigned int index);
			VkImageView GetCurrentBackBufferImage();
			VkDescriptorImageInfo GetRayTraceStagingBuffer();
			VkDescriptorImageInfo GetCurrentBackBufferImageInfo();
			VkSubmitInfo GetSubmitInfo();
			void SubmitQueue(unsigned int currentBuffer);
			void Present(std::vector<VkSemaphore> signal_semaphores);
			VkRenderPass* GetRenderPass();
			VkSurfaceKHR* GetSurface();
			VkSwapchainKHR GetSwapchain();
			VkSurfaceFormatKHR GetSurfaceFormat();
			VkPresentModeKHR GetSurfacePresentMode();
			void AttachGraphicsPipeline(VulkanGraphicsPipeline* pipeline, bool priority = false);
			void RemoveGraphicsPipeline(VulkanGraphicsPipeline* pipeline);

			Renderer::NativeWindowHandle* GetNativeWindowHandle();
			uint32_t GetImageCount();
			std::vector<VkImage> GetSwapchainImages();
			std::vector<VkImageView> GetSwpachainImageViews();
			std::vector<VkFramebuffer> GetSwapchainFrameBuffers();
			std::vector<VkCommandBuffer> GetCommandBuffers();
			VkSemaphore GetImageAvailableSemaphore();
			VkSemaphore GetRenderFinishedSemaphore();
			VkFormat GetSwapChainImageFormat();
			VkImage GetDepthImage();
			VkExtent2D GetSwapchainExtent();

			void FindNextImageIndex();
		private:

			void RebuildCommandBuffers();
			void CreateSwapchain();
			void DestroySwapchain();

			void InitSwapchain();
			void DeInitSwapchain();
			void CheckSwapChainSupport(VulkanSwapChainSupport & support);
			VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available_formats);
			VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR> available_present_modes);
			VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR & capabilities);

			// Swapchain images
			void InitSwapchainImages();
			void DeInitSwapchainImages();

			// Render pass
			void InitRenderPass();
			void DeInitRenderPass();

			// Command buffers
			void InitCommandBuffers();

			// Depth image
			void InitDepthImage();
			void DeInitDepthImage();

			// Raytrace temp image image
			void InitRaytracingTempImage();
			void DeInitRaytracingTempImage();

			// Frame buffer
			void InitFrameBuffer();
			void DeInitFrameBuffer();

			// Semaphores
			void InitSemaphores();
			void DeInitSemaphores();

			VulkanInstance* m_instance;
			VulkanDevice* m_device;
			VkSurfaceKHR* m_surface;
			VkSurfaceFormatKHR surface_format;
			VkPresentModeKHR present_mode;

			Renderer::NativeWindowHandle* m_window_handle;

			VkSwapchainKHR m_swap_chain;
			         
			std::vector<VkFence> m_fence;
			std::vector<VkImage> m_swap_chain_images;
			std::vector<VkImageView> m_swap_chain_image_views;
			std::vector<VkFramebuffer> m_swap_chain_framebuffers;
			VkExtent2D m_swap_chain_extent;
			VkFormat m_swap_chain_image_format;
			VkFormat m_depth_image_format;
			uint32_t m_frame_index = 0;
			uint32_t m_back_buffer_indices[2];
			uint32_t image_count;

			// Render pass
			VkRenderPass m_render_pass;

			// Command buffers
			std::vector<VkCommandBuffer> m_command_buffers;

			// Depth image
			VkImage m_depth_image;
			VkDeviceMemory m_depth_image_memory;
			VkImageView m_depth_image_view;

			// Raytrace Storage image
			VkImage m_raytrace_storage_image;
			VkDeviceMemory m_raytrace_storage_image_memory;
			VkImageView  m_raytrace_storage_image_view;

			// Semaphores
			VkSemaphore m_image_available_semaphore;
			VkSemaphore m_render_finished_semaphore;

			std::vector<VulkanGraphicsPipeline*> m_pipelines;

			VkPipelineStageFlags* m_wait_stages;

			bool m_should_rebuild_cmd;
		};
	}
}
#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>

#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanStatus.hpp>
#include <renderer/vulkan/VulkanSwapChainSupport.hpp>

#include <map>

namespace Renderer
{
	struct NativeWindowHandle;
	namespace Vulkan
	{
		class VulkanInstance;
		class VulkanDevice;
		class VulkanGraphicsPipeline;
		class VulkanDescriptorPool;
		class VulkanDescriptorSet;
		class VulkanTextureBuffer;
		class VulkanRenderer;
		class VulkanSwapchain : public VulkanStatus
		{
		public:
			VulkanSwapchain(VulkanRenderer* renderer,VulkanInstance* instance, VulkanDevice* device, VkSurfaceKHR* surface, Renderer::NativeWindowHandle* window_handle);
			~VulkanSwapchain();
			void RebuildSwapchain();
			VkImageView GetBackBufferImage(unsigned int index);
			VkDescriptorImageInfo GetRayTraceStagingBuffer();
			VkSubmitInfo GetSubmitInfo();
			void Present(std::vector<VkSemaphore> signal_semaphores,unsigned int frame_index);
			VkSurfaceKHR* GetSurface();
			VkSwapchainKHR GetSwapchain();
			VkSurfaceFormatKHR GetSurfaceFormat();
			VkPresentModeKHR GetSurfacePresentMode();

			Renderer::NativeWindowHandle* GetNativeWindowHandle();
			uint32_t GetImageCount();
			VkImage GetRaytracingScratchImage();
			std::vector<VkImage> GetSwapchainImages();
			std::vector<VkImageView> GetSwpachainImageViews();
			VkSemaphore GetImageAvailableSemaphore();
			VkSemaphore GetRenderFinishedSemaphore();
			VkFormat GetSwapChainImageFormat();
			VkExtent2D GetSwapchainExtent();

			char* GetRaytraceStorageTextureData();
			VulkanTextureBuffer* GetRayTraceStorageTexture();
		private:

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


			// Raytrace temp image image
			void InitRaytracingTempImage();
			void DeInitRaytracingTempImage();


			// Semaphores
			void InitSemaphores();
			void DeInitSemaphores();

			VulkanRenderer* m_renderer;
			VulkanInstance* m_instance;
			VulkanDevice* m_device;
			VkSurfaceKHR* m_surface;
			VkSurfaceFormatKHR surface_format;
			VkPresentModeKHR present_mode;

			Renderer::NativeWindowHandle* m_window_handle;

			VkSwapchainKHR m_swap_chain;
			         
			std::vector<VkImage> m_swap_chain_images;
			std::vector<VkImageView> m_swap_chain_image_views;
			VkExtent2D m_swap_chain_extent;
			VkFormat m_swap_chain_image_format;
			uint32_t m_back_buffer_indices[3];
			uint32_t image_count;






			// Raytrace Storage image
			//VkImage m_raytrace_storage_image;
			//VkDeviceMemory m_raytrace_storage_image_memory;
			//VkImageView  m_raytrace_storage_image_view;

			char* m_raytrace_storage_texture_data;
			VulkanTextureBuffer* m_raytrace_storage_texture;

			// Semaphores
			VkSemaphore m_image_available_semaphore;
			VkSemaphore m_render_finished_semaphore;

			VkPipelineStageFlags* m_wait_stages;
		};
	}
}
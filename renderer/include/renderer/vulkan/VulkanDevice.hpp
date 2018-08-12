#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>

#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanStatus.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanInstance;
		class VulkanPhysicalDevice;
		class VulkanDevice : public VulkanStatus
		{
		public:
			VulkanDevice(VulkanInstance* instance, VulkanPhysicalDevice * physical_device);
			~VulkanDevice();
			VkDevice* GetVulkanDevice();
			VkQueue* GetGraphicsQueue();
			VkQueue* GetPresentQueue();
			VkQueue* GetComputeQueue();
			VkCommandPool* GetGraphicsCommandPool();
			VkCommandPool* GetComputeCommandPool();
			void GetGraphicsCommand(VkCommandBuffer* buffers, uint32_t count);
			void GetGraphicsCommand(VkCommandBuffer* buffers, bool begin = false);
			void SubmitGraphicsCommand(VkCommandBuffer* buffers, uint32_t count);
			void FreeGraphicsCommand(VkCommandBuffer* buffers, uint32_t count);
		private:
			VulkanInstance * m_instance;
			VulkanPhysicalDevice * m_physical_device;
			VkDevice m_device = VK_NULL_HANDLE;
			VkQueue m_graphics_queue;
			VkQueue m_present_queue;
			VkQueue m_compute_queue;
			VkCommandPool m_graphics_command_pool;
			VkCommandPool m_compute_command_pool;
		};
	}
}
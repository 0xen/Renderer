#pragma once

#include <renderer/vulkan/VulkanStatus.hpp>

#include <vector>
#include <map>
#include <glm\glm.hpp>

namespace Renderer
{
	class NativeWindowHandle;
	namespace Vulkan
	{
		class VulkanInstance;
		enum VulkanFlags;
		enum ModelPoolUsage;
		class VulkanPhysicalDevice;
		class VulkanDevice;
		class VulkanSwapchain;
		class VulkanDescriptor;
		class VulkanGraphicsPipeline;
		class VulkanVertexBuffer;
		class VulkanIndexBuffer;
		class VulkanTextureBuffer;
		class VulkanRaytracePipeline;
		class VulkanUniformBuffer;
		class VulkanAcceleration;
		class VulkanComputePipeline;
		class VulkanComputeProgram;
		class VulkanModelPool;
		class VulkanDescriptorPool;
		class VulkanRenderPass;
		enum BufferChain;

		class VulkanRenderer : public VulkanStatus
		{
		public:
			VulkanRenderer();
			~VulkanRenderer();
			bool Start(Renderer::NativeWindowHandle* window_handle, unsigned int flags);
			virtual bool Start(Renderer::NativeWindowHandle* window_handle);

			void Stop();
			void Rebuild();
			VulkanUniformBuffer* CreateUniformBuffer(void* dataPtr, BufferChain level, unsigned int indexSize, unsigned int elementCount, bool modifiable);

			VulkanVertexBuffer* CreateVertexBuffer(void* dataPtr, unsigned int indexSize, unsigned int elementCount);

			VulkanIndexBuffer* CreateIndexBuffer(void* dataPtr, unsigned int indexSize, unsigned int elementCount);

			VulkanGraphicsPipeline* CreateGraphicsPipeline(VulkanRenderPass* render_pass, std::vector<std::pair<VkShaderStageFlagBits, const char*>> paths);

			VulkanComputePipeline* CreateComputePipeline(const char* path, unsigned int x, unsigned int y, unsigned int z);

			VulkanComputeProgram* CreateComputeProgram();

			VulkanModelPool* CreateModelPool(VulkanVertexBuffer* vertex_buffer, unsigned int vertex_offset, unsigned int vertex_size, VulkanIndexBuffer* index_buffer, unsigned int index_offset, unsigned int index_size, ModelPoolUsage usage);

			VulkanModelPool* CreateModelPool(VulkanVertexBuffer* vertex_buffer, unsigned int vertex_offset, unsigned int vertex_size,ModelPoolUsage usage);

			VulkanTextureBuffer* CreateTextureBuffer(void* dataPtr, VkFormat format, unsigned int width, unsigned int height, VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

			VulkanTextureBuffer* CreateTextureBuffer(void* dataPtr, BufferChain level, VkFormat format, unsigned int width, unsigned int height, VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

			VulkanTextureBuffer* CreateTextureBuffer(VkFormat format, unsigned int width, unsigned int height, VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

			VulkanTextureBuffer* CreateTextureBuffer(BufferChain level, VkFormat format, unsigned int width, unsigned int height, VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

			VulkanDescriptor* CreateDescriptor(VkDescriptorType descriptor_type, VkShaderStageFlags shader_stage, unsigned int binding,unsigned int count = 1);

			VulkanDescriptorPool* CreateDescriptorPool(std::vector<VulkanDescriptor*> descriptors);

			VulkanRaytracePipeline* CreateRaytracePipeline(VulkanRenderPass* render_pass, std::vector<std::pair<VkShaderStageFlagBits, const char*>> paths, std::vector<std::map<VkShaderStageFlagBits, const char*>> hitgroups);

			VulkanRenderPass* CreateRenderPass(unsigned int subpass_count);

			VulkanAcceleration* CreateAcceleration();

			VulkanSwapchain* GetSwapchain();

			bool IsRunning();
		private:
			void CreateSurface(Renderer::NativeWindowHandle* window_handle);
			Renderer::NativeWindowHandle* m_window_handle;
			VulkanInstance * m_instance;
			VkSurfaceKHR m_surface;
			VulkanPhysicalDevice* m_physical_device;
			VulkanDevice* m_device;
			VulkanSwapchain* m_swapchain;

			bool m_running = false;
		};
	}
}
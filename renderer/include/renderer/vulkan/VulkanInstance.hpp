#pragma once

#include <renderer/vulkan/VulkanHeader.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanStatus.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanInstance : public VulkanStatus
		{
		public:
			VulkanInstance();
			~VulkanInstance();
			VkInstance * GetInstance();
		private:
			void SetupLayersAndExtensions();
			void InitVulkanInstance();
			void DeInitVulkanInstance();
			bool CheckLayersSupport();
			std::vector<const char*> m_instance_extensions;
			std::vector<const char*> m_instance_layers;
			const uint32_t m_engine_version = VK_MAKE_VERSION(1, 0, 0);			// Engine version
			const char* m_engine_name = "Renderer";								// Engine name
			const uint32_t m_api_version = VK_MAKE_VERSION(1, 0, 68);				// Required API version number
			VkInstance m_instance;
		};
	}
}
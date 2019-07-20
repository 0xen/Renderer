#pragma once

#include <renderer\vulkan\VulkanPipeline.hpp>
#include <renderer\vulkan\VulkanStatus.hpp>

namespace Renderer
{
	namespace Vulkan
	{
		class VulkanComputePipeline : public VulkanPipeline, public VulkanStatus
		{
		public:
			VulkanComputePipeline(VulkanDevice * device, const char* path, unsigned int x, unsigned int y, unsigned int z);
			virtual ~VulkanComputePipeline();
			virtual bool Build();
			virtual bool CreatePipeline();
			virtual void DestroyPipeline();
			virtual void AttachToCommandBuffer(VkCommandBuffer & command_buffer);
			unsigned int GetX();
			unsigned int GetY();
			unsigned int GetZ();
			void SetX(unsigned int x);
			void SetY(unsigned int y);
			void SetZ(unsigned int z);
		private:
			VkShaderModule m_shader_module;
			unsigned int m_x;
			unsigned int m_y;
			unsigned int m_z;
		};
	}
}
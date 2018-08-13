#pragma once

#include <renderer/vulkan/VulkanBuffer.hpp>
#include <renderer/vulkan/VulkanBufferDescriptor.hpp>
#include <renderer/IUniformBuffer.hpp>

namespace Renderer
{

	namespace Vulkan
	{
		class VulkanUniformBuffer : public IUniformBuffer, public VulkanBuffer, public VulkanBufferDescriptor
		{
		public:
			VulkanUniformBuffer(VulkanDevice* device, void* dataPtr, unsigned int indexSize, unsigned int elementCount, DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding);
			~VulkanUniformBuffer();
		};
	}
}
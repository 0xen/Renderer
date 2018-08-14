#include <renderer/vulkan/VulkanComputeProgram.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>
#include <renderer/vulkan/VulkanComputePipeline.hpp>

Renderer::Vulkan::VulkanComputeProgram::VulkanComputeProgram(VulkanDevice * device)
{
	m_device = device;

	VkFenceCreateInfo fenceCreateInfo = VulkanInitializers::CreateFenceInfo();
	ErrorCheck(vkCreateFence(*device->GetVulkanDevice(), &fenceCreateInfo, NULL, &m_fence));
}

Renderer::Vulkan::VulkanComputeProgram::~VulkanComputeProgram()
{
	vkDestroyFence(*m_device->GetVulkanDevice(), m_fence, NULL);
}

void Renderer::Vulkan::VulkanComputeProgram::Build()
{
	m_command_buffer = VulkanCommon::BeginSingleTimeCommands(m_device, *m_device->GetGraphicsCommandPool());
	for (auto cp : m_pipelines)
	{
		VulkanComputePipeline* vcp = dynamic_cast<VulkanComputePipeline*>(cp);
		vcp->AttachToCommandBuffer(m_command_buffer);
	}

	ErrorCheck(vkEndCommandBuffer(
		m_command_buffer
	));
}

void Renderer::Vulkan::VulkanComputeProgram::Run()
{
	VkSubmitInfo submitInfo = VulkanInitializers::SubmitInfo(m_command_buffer);

	ErrorCheck(vkQueueSubmit(*m_device->GetComputeQueue(), 1, &submitInfo, m_fence));
	ErrorCheck(vkWaitForFences(*m_device->GetVulkanDevice(), 1, &m_fence, VK_TRUE, LONG_MAX));
}

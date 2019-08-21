#include <renderer/vulkan/VulkanComputePipeline.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanUniformBuffer.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>
#include <renderer/vulkan/VulkanDescriptorPool.hpp>
#include <renderer/vulkan/VulkanDescriptorSet.hpp>


Renderer::Vulkan::VulkanComputePipeline::VulkanComputePipeline(VulkanDevice * device, const char * path, unsigned int x, unsigned int y, unsigned int z) :
	VulkanPipeline(device, { { VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT, path } })
{
	m_x = x;
	m_y = y;
	m_z = z;
}

Renderer::Vulkan::VulkanComputePipeline::~VulkanComputePipeline()
{
	vkDestroyPipeline(
		*m_device->GetVulkanDevice(),
		m_pipeline,
		nullptr
	);

	vkDestroyPipelineLayout(
		*m_device->GetVulkanDevice(),
		m_pipeline_layout,
		nullptr
	);

	vkDestroyShaderModule(
		*m_device->GetVulkanDevice(),
		m_shader_module,
		nullptr
	);
	
}

bool Renderer::Vulkan::VulkanComputePipeline::Build()
{
	return Rebuild();
}

bool Renderer::Vulkan::VulkanComputePipeline::CreatePipeline()
{

	std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
	descriptor_set_layouts.resize(m_descriptor_pools.size());
	for (auto descriptor_pool : m_descriptor_pools)
	{
		descriptor_set_layouts[descriptor_pool.first] = descriptor_pool.second->GetDescriptorSetLayout();
	}

	VkPipelineLayoutCreateInfo pipeline_layout_info = VulkanInitializers::PipelineLayoutCreateInfo(descriptor_set_layouts);
	ErrorCheck(vkCreatePipelineLayout(
		*m_device->GetVulkanDevice(),
		&pipeline_layout_info,
		nullptr,
		&m_pipeline_layout
	));

	if (HasError())return false;

	auto shaders = GetPaths();

	if (shaders.size() <= 0 || shaders[0].first != VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT) return false;

	std::vector<char> shaderCode = VulkanCommon::ReadFile(shaders[0].second);
	m_shader_module = VulkanCommon::CreateShaderModule(m_device, shaderCode);

	VkPipelineShaderStageCreateInfo shader_info = VulkanInitializers::PipelineShaderStageCreateInfo(m_shader_module, "main", VK_SHADER_STAGE_COMPUTE_BIT);
	VkComputePipelineCreateInfo compute_pipeline_create_info = VulkanInitializers::ComputePipelineCreateInfo(m_pipeline_layout, shader_info);

	ErrorCheck(vkCreateComputePipelines(
		*m_device->GetVulkanDevice(),
		0,
		1,
		&compute_pipeline_create_info,
		nullptr,
		&m_pipeline
	));

	if (HasError())return false;

	return true;
}

void Renderer::Vulkan::VulkanComputePipeline::DestroyPipeline()
{
	vkDestroyPipelineLayout(
		*m_device->GetVulkanDevice(),
		m_pipeline_layout,
		nullptr
	);
	vkDestroyPipeline(
		*m_device->GetVulkanDevice(),
		m_pipeline,
		nullptr
	);
	vkDestroyShaderModule(
		*m_device->GetVulkanDevice(),
		m_shader_module,
		nullptr
	);
}

void Renderer::Vulkan::VulkanComputePipeline::AttachToCommandBuffer(VkCommandBuffer & command_buffer)
{
	vkCmdBindPipeline(
		command_buffer,
		VK_PIPELINE_BIND_POINT_COMPUTE,
		m_pipeline
	);
	for (auto it = m_descriptor_sets.begin(); it != m_descriptor_sets.end(); it++)
	{
		vkCmdBindDescriptorSets(
			command_buffer,
			VK_PIPELINE_BIND_POINT_COMPUTE,
			m_pipeline_layout,
			it->first,
			1,
			&it->second->GetDescriptorSet(),
			0,
			0
		);
	}

	vkCmdDispatch(
		command_buffer,
		GetX(),
		GetY(),
		GetZ()
	);
}

unsigned int Renderer::Vulkan::VulkanComputePipeline::GetX()
{
	return m_x;
}

unsigned int Renderer::Vulkan::VulkanComputePipeline::GetY()
{
	return m_y;
}

unsigned int Renderer::Vulkan::VulkanComputePipeline::GetZ()
{
	return m_z;
}

void Renderer::Vulkan::VulkanComputePipeline::SetX(unsigned int x)
{
	m_x = x;
}

void Renderer::Vulkan::VulkanComputePipeline::SetY(unsigned int y)
{
	m_y = y;
}

void Renderer::Vulkan::VulkanComputePipeline::SetZ(unsigned int z)
{
	m_z = z;
}

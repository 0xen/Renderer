#include <renderer/vulkan/VulkanComputePipeline.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanUniformBuffer.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>
#include <renderer/vulkan/VulkanDescriptorPool.hpp>
#include <renderer/vulkan/VulkanDescriptorSet.hpp>


Renderer::Vulkan::VulkanComputePipeline::VulkanComputePipeline(VulkanDevice * device, const char * path, unsigned int x, unsigned int y, unsigned int z) :
	IComputePipeline(path,x,y,z),
	VulkanPipeline(device, { { ShaderStage::COMPUTE_SHADER, path } }),
	IPipeline({ { ShaderStage::COMPUTE_SHADER, path } })
{

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
	for (auto descriptor_pool : m_descriptor_pools)
	{
		descriptor_set_layouts.push_back(descriptor_pool->GetDescriptorSetLayout());
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

	if (shaders.size() <= 0 || shaders[0].first != ShaderStage::COMPUTE_SHADER) return false;

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

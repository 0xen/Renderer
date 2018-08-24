#include <renderer/vulkan/VulkanComputePipeline.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanUniformBuffer.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>


Renderer::Vulkan::VulkanComputePipeline::VulkanComputePipeline(VulkanDevice * device, const char * path, unsigned int x, unsigned int y, unsigned int z) :
	IComputePipeline(path,x,y,z),
	VulkanPipeline(device, { { ShaderStage::COMPUTE_SHADER, path } }),
	IPipeline({ { ShaderStage::COMPUTE_SHADER, path } })
{

}

Renderer::Vulkan::VulkanComputePipeline::~VulkanComputePipeline()
{
}

bool Renderer::Vulkan::VulkanComputePipeline::Build()
{
	return Rebuild();
}

bool Renderer::Vulkan::VulkanComputePipeline::CreatePipeline()
{
	m_descriptor_pool_sizes.clear();
	m_layout_bindings.clear();

	for (VulkanBufferDescriptor* descriptor : m_descriptor)
	{
		m_descriptor_pool_sizes.push_back(VulkanInitializers::DescriptorPoolSize(descriptor->GetVulkanDescriptorType()));
		m_layout_bindings.push_back(VulkanInitializers::DescriptorSetLayoutBinding(descriptor->GetVulkanDescriptorType(), descriptor->GetVulkanShaderStage(), descriptor->GetBinding()));
	}

	VkDescriptorPoolCreateInfo descriptor_pool_create_info = VulkanInitializers::DescriptorPoolCreateInfo(m_descriptor_pool_sizes, 1);
	ErrorCheck(vkCreateDescriptorPool(
		*m_device->GetVulkanDevice(),
		&descriptor_pool_create_info,
		nullptr,
		&m_descriptor_pool
	));

	if (HasError())return false;

	VkDescriptorSetLayoutCreateInfo layout_info = VulkanInitializers::DescriptorSetLayoutCreateInfo(m_layout_bindings);
	ErrorCheck(vkCreateDescriptorSetLayout(
		*m_device->GetVulkanDevice(),
		&layout_info,
		nullptr,
		&m_descriptor_set_layout
	));

	if (HasError())return false;

	VkPipelineLayoutCreateInfo pipeline_layout_info = VulkanInitializers::PipelineLayoutCreateInfo(m_descriptor_set_layout);
	ErrorCheck(vkCreatePipelineLayout(
		*m_device->GetVulkanDevice(),
		&pipeline_layout_info,
		nullptr,
		&m_pipeline_layout
	));

	if (HasError())return false;

	m_descriptor_set_layouts.clear();
	m_descriptor_set_layouts.push_back(m_descriptor_set_layout);
	VkDescriptorSetAllocateInfo descriptor_set_alloc_info = VulkanInitializers::DescriptorSetAllocateInfo(m_descriptor_set_layouts, m_descriptor_pool);


	ErrorCheck(vkAllocateDescriptorSets(
		*m_device->GetVulkanDevice(),
		&descriptor_set_alloc_info,
		&m_descriptor_set
	));

	if (HasError())return false;

	auto shaders = GetPaths();

	if (shaders[ShaderStage::COMPUTE_SHADER] == nullptr) return false;

	std::vector<char> shaderCode = VulkanCommon::ReadFile(shaders[ShaderStage::COMPUTE_SHADER]);

	auto shader_module = VulkanCommon::CreateShaderModule(m_device, shaderCode);

	VkPipelineShaderStageCreateInfo shader_info = VulkanInitializers::PipelineShaderStageCreateInfo(shader_module, "main", VK_SHADER_STAGE_COMPUTE_BIT);
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

	VkDeviceSize offset = 0;
	m_write_descriptor_sets.clear();
	for (VulkanBufferDescriptor* descriptor : m_descriptor)
	{
		if (descriptor->GetDescriptorType() == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
		{
			m_write_descriptor_sets.push_back(VulkanInitializers::WriteDescriptorSet(m_descriptor_set, descriptor->GetDescriptorImageInfo(), descriptor->GetVulkanDescriptorType(), descriptor->GetBinding()));
		}
		else
		{
			m_write_descriptor_sets.push_back(VulkanInitializers::WriteDescriptorSet(m_descriptor_set, descriptor->GetDescriptorBufferInfo(), descriptor->GetVulkanDescriptorType(), descriptor->GetBinding()));
		}
	}
	vkUpdateDescriptorSets(*m_device->GetVulkanDevice(), (uint32_t)m_write_descriptor_sets.size(), m_write_descriptor_sets.data(), 0, NULL);

	return true;
}

void Renderer::Vulkan::VulkanComputePipeline::DestroyPipeline()
{
	vkDestroyPipelineLayout(*m_device->GetVulkanDevice(), m_pipeline_layout, nullptr);
	vkDestroyPipeline(*m_device->GetVulkanDevice(), m_pipeline, nullptr);
}

void Renderer::Vulkan::VulkanComputePipeline::AttachToCommandBuffer(VkCommandBuffer & command_buffer)
{
	vkCmdBindPipeline(
		command_buffer,
		VK_PIPELINE_BIND_POINT_COMPUTE,
		m_pipeline
	);
	vkCmdBindDescriptorSets(
		command_buffer,
		VK_PIPELINE_BIND_POINT_COMPUTE,
		m_pipeline_layout,
		0,
		1,
		&m_descriptor_set,
		0,
		0
	);
	vkCmdDispatch(
		command_buffer,
		GetX(),
		GetY(),
		GetZ()
	);
}

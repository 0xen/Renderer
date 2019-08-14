#include <renderer/vulkan/VulkanGraphicsPipeline.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanUniformBuffer.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanSwapchain.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>
#include <renderer/vulkan/VulkanModelPool.hpp>
#include <renderer/vulkan/VulkanDescriptorPool.hpp>
#include <renderer/vulkan/VulkanDescriptorSet.hpp>
#include <renderer/VertexBase.hpp>

#include <glm/glm.hpp>

#include <map>

using namespace Renderer;
using namespace Renderer::Vulkan;

Renderer::Vulkan::VulkanGraphicsPipeline::VulkanGraphicsPipeline(VulkanDevice * device, VulkanSwapchain* swapchain, std::vector<std::pair<VkShaderStageFlagBits, const char*>> paths) :
	VulkanPipeline(device, paths)
{
	m_swapchain = swapchain;

	m_change = false;
	InitPipelineCreateInfo();
}

Renderer::Vulkan::VulkanGraphicsPipeline::~VulkanGraphicsPipeline()
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
	for (int i = 0; i < m_shader_stages.size(); i++)
	{
		vkDestroyShaderModule(
			*m_device->GetVulkanDevice(),
			m_shader_stages[i].module,
			nullptr
		);
	}

}

bool Renderer::Vulkan::VulkanGraphicsPipeline::Build()
{
	return Rebuild();
}

bool Renderer::Vulkan::VulkanGraphicsPipeline::CreatePipeline()
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



	for (auto shader = shaders.begin(); shader != shaders.end(); shader++)
	{
		std::vector<char> shaderCode = VulkanCommon::ReadFile(shader->second);

		auto shader_module = VulkanCommon::CreateShaderModule(m_device, shaderCode);

		m_shader_stages.push_back(VulkanInitializers::PipelineShaderStageCreateInfo(shader_module, "main", shader->first));
	}

	m_binding_descriptions.clear();

	for (auto base : m_vertex_bases)
	{
		m_binding_descriptions.push_back(VulkanInitializers::VertexInputBinding(base.binding, base.size, base.vertex_input_rate));
	}

	//m_binding_descriptions.push_back(VulkanInitializers::VertexInputBinding(1, sizeof(glm::mat4), VK_VERTEX_INPUT_RATE_INSTANCE));


	m_attribute_descriptions.clear();

	for (auto base : m_vertex_bases)
	{
		for (auto vertex = base.vertex_bindings.begin(); vertex != base.vertex_bindings.end(); vertex++)
		{
			switch (vertex->GetFormat())
			{
			case VkFormat::VK_FORMAT_R32G32B32A32_SFLOAT://DataFormat::MAT4_FLOAT:
			{
				uint32_t location = vertex->GetLocation();
				unsigned int size_of = sizeof(glm::vec4);
				m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(base.binding, location, vertex->GetFormat(), vertex->GetOffset() + 0));
				m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(base.binding, location + 1, vertex->GetFormat(), vertex->GetOffset() + size_of));
				m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(base.binding, location + 2, vertex->GetFormat(), vertex->GetOffset() + (2 * size_of)));
				m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(base.binding, location + 3, vertex->GetFormat(), vertex->GetOffset() + (3 * size_of)));
			}
			break;
			case VkFormat::VK_FORMAT_R32_SINT:
			{
				uint32_t location = vertex->GetLocation();
				unsigned int size_of = sizeof(short);
				m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(base.binding, location, VK_FORMAT_R16_SINT, vertex->GetOffset() + 0));
				m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(base.binding, location + 1, VK_FORMAT_R16_SINT, vertex->GetOffset() + size_of));
			}
			break;
			default:
				m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(base.binding, vertex->GetLocation(), vertex->GetFormat(), vertex->GetOffset()));
				break;
			}
			
		}
	}

	VkPipelineVertexInputStateCreateInfo vertex_input_info = VulkanInitializers::PipelineVertexInputStateCreateInfo(m_binding_descriptions, m_attribute_descriptions);

	// Viewport state
	VkPipelineViewportStateCreateInfo viewport_state = VulkanInitializers::PipelineViewportStateCreateInfo(1, 1);

	// Rasteriser
	// Needs to be abstracted
	VkPipelineRasterizationStateCreateInfo rasterizer = VulkanInitializers::PipelineRasterizationStateCreateInfo(
		m_graphics_pipeline_config.culling,
		m_graphics_pipeline_config.front_face,
		m_graphics_pipeline_config.polygon_mode,
		1.0f);

	// Multi sampling
	VkPipelineMultisampleStateCreateInfo multisampling = VulkanInitializers::PipelineMultisampleStateCreateInfo();

	// Depth stencil
	VkPipelineDepthStencilStateCreateInfo depth_stencil = VulkanInitializers::PipelineDepthStencilStateCreateInfo(m_graphics_pipeline_config.use_depth_stencil);


	
	VkPipelineDynamicStateCreateInfo dynamic_states_info = VulkanInitializers::PipelineDynamicStateCreateInfo(m_graphics_pipeline_config.dynamic_states);


	// Color blending
	VkPipelineColorBlendAttachmentState color_blend_attachment = VulkanInitializers::PipelineColorBlendAttachmentState();
	VkPipelineColorBlendStateCreateInfo color_blending = VulkanInitializers::PipelineColorBlendStateCreateInfo(color_blend_attachment);

	// Triangle pipeline
	VkPipelineInputAssemblyStateCreateInfo input_assembly = VulkanInitializers::PipelineInputAssemblyStateCreateInfo(m_graphics_pipeline_config.topology);

	VkGraphicsPipelineCreateInfo pipeline_info = VulkanInitializers::GraphicsPipelineCreateInfo(m_shader_stages, vertex_input_info, input_assembly,
		viewport_state, rasterizer, multisampling, color_blending, depth_stencil, m_pipeline_layout, *m_swapchain->GetRenderPass(), dynamic_states_info);

	// If we support it, define that this pipeline has derivatives
	if (m_graphics_pipeline_config.allow_darivatives) 
		pipeline_info.flags |= VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT;
	// If we have a parent and that parent allows for derivatives, continue
	if (m_graphics_pipeline_config.parent != nullptr)
	{
		assert(m_graphics_pipeline_config.parent->m_graphics_pipeline_config.allow_darivatives && "Parent dose not support derivatives");
		pipeline_info.flags |= VK_PIPELINE_CREATE_DERIVATIVE_BIT;

		pipeline_info.basePipelineHandle = m_graphics_pipeline_config.parent->m_pipeline;

		// If VK_PIPELINE_CREATE_DERIVATIVE_BIT is set and we have a valid pipeline, basePipelineIndex must be -1 https://www.khronos.org/registry/vulkan/specs/1.1-extensions/man/html/VkGraphicsPipelineCreateInfo.html
		pipeline_info.basePipelineIndex = -1;
	}

	ErrorCheck(vkCreateGraphicsPipelines(
		*m_device->GetVulkanDevice(),
		VK_NULL_HANDLE,
		1,
		&pipeline_info,
		nullptr,
		&m_pipeline
	));

	if (HasError())return false;

	return true;
}

void Renderer::Vulkan::VulkanGraphicsPipeline::DestroyPipeline()
{
	vkDestroyPipelineLayout(*m_device->GetVulkanDevice(), m_pipeline_layout, nullptr);
	vkDestroyPipeline(*m_device->GetVulkanDevice(), m_pipeline, nullptr);
}

void Renderer::Vulkan::VulkanGraphicsPipeline::AttachToCommandBuffer(VkCommandBuffer & command_buffer)
{
	if (m_model_pools.size() == 0)return;
	vkCmdBindPipeline(
		command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS,
		m_pipeline
	);
	for(auto it = m_descriptor_sets.begin(); it!= m_descriptor_sets.end(); it++)
	{
		vkCmdBindDescriptorSets(
			command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			m_pipeline_layout,
			it->first,
			1,
			&it->second->GetDescriptorSet(),
			0,
			NULL
		);
	}
	for (auto model_pool : m_model_pools)
	{
		model_pool->AttachToCommandBuffer(command_buffer,this);
	}
}

void Renderer::Vulkan::VulkanGraphicsPipeline::AttachModelPool(VulkanModelPool * model_pool)
{
	m_model_pools.push_back(model_pool);
}

void Renderer::Vulkan::VulkanGraphicsPipeline::AttachVertexBinding(VertexBase vertex_binding)
{
	m_vertex_bases.push_back(vertex_binding);
}

bool Renderer::Vulkan::VulkanGraphicsPipeline::HasChanged()
{
	if (m_change)
	{
		m_change = false;
		return true;
	}

	for (auto pool : m_model_pools)
	{
		if (pool->HasChanged())return true;
	}
	return false;
}

VulkanGraphicsPipelineConfig & Renderer::Vulkan::VulkanGraphicsPipeline::GetGraphicsPipelineConfig()
{
	return m_graphics_pipeline_config;
}

void Renderer::Vulkan::VulkanGraphicsPipeline::InitPipelineCreateInfo()
{
	// Dynamic shader stages
	m_graphics_pipeline_config.dynamic_states = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
		VK_DYNAMIC_STATE_LINE_WIDTH,
	};
	m_graphics_pipeline_config.parent = nullptr;

	m_graphics_pipeline_config.use_depth_stencil = true;
	m_graphics_pipeline_config.allow_darivatives = false;

	m_graphics_pipeline_config.culling = VK_CULL_MODE_BACK_BIT;
	m_graphics_pipeline_config.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	m_graphics_pipeline_config.front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	m_graphics_pipeline_config.polygon_mode = VK_POLYGON_MODE_FILL;
}

#include <renderer/vulkan/VulkanGraphicsPipeline.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanUniformBuffer.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanSwapchain.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>
#include <renderer/vulkan/VulkanModelPool.hpp>
#include <renderer/vulkan/VulkanDescriptorPool.hpp>
#include <renderer/vulkan/VulkanDescriptorSet.hpp>
#include <renderer/ShaderStage.hpp>
#include <renderer/DataFormat.hpp>
#include <renderer/IModelPool.hpp>

#include <glm/glm.hpp>

#include <map>

using namespace Renderer;
using namespace Renderer::Vulkan;

std::map<Renderer::ShaderStage, VkShaderStageFlagBits> Renderer::Vulkan::VulkanGraphicsPipeline::m_shader_stage_flags
{
{ Renderer::ShaderStage::COMPUTE_SHADER , VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT },
{ Renderer::ShaderStage::FRAGMENT_SHADER , VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT },
{ Renderer::ShaderStage::VERTEX_SHADER , VkShaderStageFlagBits::VK_SHADER_STAGE_VERTEX_BIT },
{ Renderer::ShaderStage::GEOMETRY_SHADER , VkShaderStageFlagBits::VK_SHADER_STAGE_GEOMETRY_BIT },
};


std::map<Renderer::DataFormat, VkFormat> Renderer::Vulkan::VulkanGraphicsPipeline::m_formats
{
	{ Renderer::DataFormat::R32G32_FLOAT,VkFormat::VK_FORMAT_R32G32_SFLOAT },
	{ Renderer::DataFormat::R32G32B32_FLOAT,VkFormat::VK_FORMAT_R32G32B32_SFLOAT },
	{ Renderer::DataFormat::R8G8B8A8_UNORM,VkFormat::VK_FORMAT_R8G8B8A8_UNORM },
};


std::map<Renderer::VertexInputRate, VkVertexInputRate> Renderer::Vulkan::VulkanGraphicsPipeline::m_vertex_input_rates
{
	{ Renderer::VertexInputRate::INPUT_RATE_INSTANCE,VkVertexInputRate::VK_VERTEX_INPUT_RATE_INSTANCE },
	{ Renderer::VertexInputRate::INPUT_RATE_VERTEX,VkVertexInputRate::VK_VERTEX_INPUT_RATE_VERTEX },
};

Renderer::Vulkan::VulkanGraphicsPipeline::VulkanGraphicsPipeline(VulkanDevice * device, VulkanSwapchain* swapchain, std::map<ShaderStage, const char*> paths) :
	IGraphicsPipeline(paths),
	VulkanPipeline(device, paths),
	IPipeline(paths)
{
	m_swapchain = swapchain;

	m_change = false;
}

Renderer::Vulkan::VulkanGraphicsPipeline::~VulkanGraphicsPipeline()
{
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


	std::vector<VkPipelineShaderStageCreateInfo> shader_stages;

	for (auto shader = shaders.begin(); shader != shaders.end(); shader++)
	{
		std::vector<char> shaderCode = VulkanCommon::ReadFile(shader->second);

		auto shader_module = VulkanCommon::CreateShaderModule(m_device, shaderCode);

		shader_stages.push_back(VulkanInitializers::PipelineShaderStageCreateInfo(shader_module, "main", GetShaderStageFlag(shader->first)));
	}

	m_binding_descriptions.clear();

	for (auto base : m_vertex_bases)
	{
		m_binding_descriptions.push_back(VulkanInitializers::VertexInputBinding(base.binding, base.size, GetVertexInputRate(base.vertex_input_rate)));
	}

	//m_binding_descriptions.push_back(VulkanInitializers::VertexInputBinding(1, sizeof(glm::mat4), VK_VERTEX_INPUT_RATE_INSTANCE));


	m_attribute_descriptions.clear();

	for (auto base : m_vertex_bases)
	{
		for (auto vertex = base.vertex_bindings.begin(); vertex != base.vertex_bindings.end(); vertex++)
		{
			switch (vertex->GetFormat())
			{
			case DataFormat::MAT4_FLOAT:
			{
				VkFormat format = VK_FORMAT_R32G32B32A32_UINT;
				uint32_t location = vertex->GetLocation();
				unsigned int size_of = sizeof(glm::vec4);
				m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(1, location, format, vertex->GetOffset() + 0));
				m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(1, location + 1, format, vertex->GetOffset() + size_of));
				m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(1, location + 2, format, vertex->GetOffset() + (2 * size_of)));
				m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(1, location + 3, format, vertex->GetOffset() + (3 * size_of)));
				//m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(0, vertex->GetLocation(), GetFormat(vertex->GetFormat()), vertex->GetOffset()));
			}
				break;
			default:
				m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(0, vertex->GetLocation(), GetFormat(vertex->GetFormat()), vertex->GetOffset()));
				break;
			}
			
		}
	}

	// Needs to be replaced with abstract camera in future
	/*m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(1, 3, VK_FORMAT_R32G32B32A32_UINT, 0));
	m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(1, 4, VK_FORMAT_R32G32B32A32_UINT, sizeof(glm::vec4)));
	m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(1, 5, VK_FORMAT_R32G32B32A32_UINT, 2 * sizeof(glm::vec4)));
	m_attribute_descriptions.push_back(VulkanInitializers::VertexInputAttributeDescription(1, 6, VK_FORMAT_R32G32B32A32_UINT, 3 * sizeof(glm::vec4)));*/

	VkPipelineVertexInputStateCreateInfo vertex_input_info = VulkanInitializers::PipelineVertexInputStateCreateInfo(m_binding_descriptions, m_attribute_descriptions);

	// Viewport state
	VkPipelineViewportStateCreateInfo viewport_state = VulkanInitializers::PipelineViewportStateCreateInfo(1, 1);

	// Rasteriser
	// Needs to be abstracted
	VkPipelineRasterizationStateCreateInfo rasterizer = VulkanInitializers::PipelineRasterizationStateCreateInfo(
		VkCullModeFlagBits::VK_CULL_MODE_NONE,
		VK_FRONT_FACE_COUNTER_CLOCKWISE,
		VkPolygonMode::VK_POLYGON_MODE_LINE,
		1.0f);

	// Multi sampling
	VkPipelineMultisampleStateCreateInfo multisampling = VulkanInitializers::PipelineMultisampleStateCreateInfo();

	// Depth stencil
	VkPipelineDepthStencilStateCreateInfo depth_stencil = VulkanInitializers::PipelineDepthStencilStateCreateInfo(m_use_depth_stencil);

	// Color blending
	VkPipelineColorBlendAttachmentState color_blend_attachment = VulkanInitializers::PipelineColorBlendAttachmentState();

	// Dynamic shader stages
	std::vector<VkDynamicState> dynamic_states = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
		VK_DYNAMIC_STATE_LINE_WIDTH,
	};

	VkPipelineDynamicStateCreateInfo dynamic_states_info = VulkanInitializers::PipelineDynamicStateCreateInfo(dynamic_states);

	VkPipelineColorBlendStateCreateInfo color_blending = VulkanInitializers::PipelineColorBlendStateCreateInfo(color_blend_attachment);

	// Triangle pipeline
	VkPipelineInputAssemblyStateCreateInfo input_assembly = VulkanInitializers::PipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	VkGraphicsPipelineCreateInfo pipeline_info = VulkanInitializers::GraphicsPipelineCreateInfo(shader_stages, vertex_input_info, input_assembly,
		viewport_state, rasterizer, multisampling, color_blending, depth_stencil, m_pipeline_layout, *m_swapchain->GetRenderPass(), dynamic_states_info);

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

void Renderer::Vulkan::VulkanGraphicsPipeline::AttachModelPool(IModelPool * model_pool)
{
	m_model_pools.push_back(dynamic_cast<VulkanModelPool*>(model_pool));
}

void Renderer::Vulkan::VulkanGraphicsPipeline::AttachVertexBinding(VertexBase vertex_binding)
{
	m_vertex_bases.push_back(vertex_binding);
}

void Renderer::Vulkan::VulkanGraphicsPipeline::UseDepth(bool depth)
{
	m_use_depth_stencil = depth;
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

VkShaderStageFlagBits Renderer::Vulkan::VulkanGraphicsPipeline::GetShaderStageFlag(ShaderStage stage)
{
	return m_shader_stage_flags[stage];
}

VkFormat Renderer::Vulkan::VulkanGraphicsPipeline::GetFormat(Renderer::DataFormat format)
{
	return m_formats[format];
}

VkVertexInputRate Renderer::Vulkan::VulkanGraphicsPipeline::GetVertexInputRate(Renderer::VertexInputRate input_rate)
{
	return m_vertex_input_rates[input_rate];
}
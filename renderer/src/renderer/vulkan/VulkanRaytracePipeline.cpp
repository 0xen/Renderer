#include <renderer/vulkan/VulkanRaytracePipeline.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>
#include <renderer/vulkan/VulkanSwapchain.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanDescriptorPool.hpp>


Renderer::Vulkan::VulkanRaytracePipeline::VulkanRaytracePipeline(VulkanDevice * device, VulkanSwapchain * swapchain, std::map<ShaderStage, const char*> paths, std::vector<std::map<ShaderStage, const char*>> hitgroups) :
	IPipeline(paths),
	VulkanGraphicsPipeline(device, swapchain, paths)
{
	m_device = device;
	m_swapchain = swapchain;
	m_hitgroups = hitgroups;
}

Renderer::Vulkan::VulkanRaytracePipeline::~VulkanRaytracePipeline()
{
}

bool Renderer::Vulkan::VulkanRaytracePipeline::Build()
{
	return Rebuild();
}

bool Renderer::Vulkan::VulkanRaytracePipeline::CreatePipeline()
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
		// Get the shader src binery
		std::vector<char> shaderCode = VulkanCommon::ReadFile(shader->second);
		// Create a module
		auto shader_module = VulkanCommon::CreateShaderModule(m_device, shaderCode);
		// Push the shader back
		VkShaderStageFlagBits shaderStage = GetShaderStageFlag(shader->first);
		m_shader_stages.push_back(VulkanInitializers::PipelineShaderStageCreateInfo(shader_module, "main", shaderStage));

		uint32_t shaderID = m_shader_stages.size() - 1;


		VkRayTracingShaderGroupCreateInfoNV group = VulkanInitializers::RayTracingShaderGroupCreateNV(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV);
		group.generalShader = shaderID;
		m_shader_groups.push_back(group);
		m_shader_groupIndex++;
	}

	for (auto& hitgroup : m_hitgroups)
	{


		VkRayTracingShaderGroupCreateInfoNV group = VulkanInitializers::RayTracingShaderGroupCreateNV(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV);

		for (auto shader = hitgroup.begin(); shader != hitgroup.end(); shader++)
		{
			// Get the shader src binery
			std::vector<char> shaderCode = VulkanCommon::ReadFile(shader->second);
			// Create a module
			auto shader_module = VulkanCommon::CreateShaderModule(m_device, shaderCode);
			// Push the shader back
			VkShaderStageFlagBits shaderStage = GetShaderStageFlag(shader->first);
			m_shader_stages.push_back(VulkanInitializers::PipelineShaderStageCreateInfo(shader_module, "main", shaderStage));

			uint32_t shaderID = m_shader_stages.size() - 1;

			switch (shaderStage)
			{
			case VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV:
				switch (shaderStage)
				{
				case VK_SHADER_STAGE_ANY_HIT_BIT_NV:
					group.anyHitShader = shaderID;
					break;

				case VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV:
					group.closestHitShader = shaderID;
					break;

				case VK_SHADER_STAGE_INTERSECTION_BIT_NV:
					group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_NV;
					group.intersectionShader = shaderID;
					break;
				}
				break;
			default:
				break;
			}
		}
		m_shader_groups.push_back(group);
		m_shader_groupIndex++;
	}

	VkRayTracingPipelineCreateInfoNV create_info = VulkanInitializers::RayTracePipelineCreateInfoNV(m_shader_stages, m_shader_groups, m_pipeline_layout, m_max_depth);
	
	ErrorCheck(vkCreateRayTracingPipelinesNV(
		*m_device->GetVulkanDevice(),
		nullptr,
		1,
		&create_info,
		nullptr, 
		&m_pipeline
	));

	if (HasError())return false;
	return true;
}

void Renderer::Vulkan::VulkanRaytracePipeline::DestroyPipeline()
{
}

void Renderer::Vulkan::VulkanRaytracePipeline::AttachToCommandBuffer(VkCommandBuffer & command_buffer)
{
}

void Renderer::Vulkan::VulkanRaytracePipeline::AttachModelPool(IModelPool * model_pool)
{
}

void Renderer::Vulkan::VulkanRaytracePipeline::AttachVertexBinding(VertexBase vertex_binding)
{
	m_vertex_bases.push_back(vertex_binding);
}

void Renderer::Vulkan::VulkanRaytracePipeline::UseDepth(bool depth)
{
}

void Renderer::Vulkan::VulkanRaytracePipeline::UseCulling(bool culling)
{
}

void Renderer::Vulkan::VulkanRaytracePipeline::DefinePrimitiveTopology(PrimitiveTopology top)
{
}

void Renderer::Vulkan::VulkanRaytracePipeline::SetMaxRecursionDepth(uint32_t max_depth)
{
	m_max_depth = max_depth;
}

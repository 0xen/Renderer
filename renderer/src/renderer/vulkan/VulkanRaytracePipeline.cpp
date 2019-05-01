#include <renderer/vulkan/VulkanRaytracePipeline.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>

Renderer::Vulkan::VulkanRaytracePipeline::VulkanRaytracePipeline(VulkanDevice * device, VulkanSwapchain * swapchain, std::map<ShaderStage, const char*> paths) :
	IPipeline(paths),
	VulkanGraphicsPipeline(device, swapchain, paths)
{
	m_device = device;
	m_swapchain = swapchain;
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


	this;
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


		if (m_recording_hit_shaders)
		{
			auto& group = m_shader_groups[m_shader_groupIndex];
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
		else if (shaderStage == VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV /* || ....*/)
		{
			m_shader_groups.push_back(VulkanInitializers::RayTracingShaderGroupCreateNV(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV));
			m_recording_hit_shaders = true;
			m_shader_groupIndex++;
		}
		else
		{
			// Create and store a shader group

			VkRayTracingShaderGroupCreateInfoNV group = VulkanInitializers::RayTracingShaderGroupCreateNV(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV);
			group.generalShader = shaderID;
			m_shader_groups.push_back(group);
			m_shader_groupIndex++;
		}
	}


	return false;
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

#include <renderer/vulkan/VulkanRaytracePipeline.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>
#include <renderer/vulkan/VulkanSwapchain.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanDescriptorPool.hpp>
#include <renderer/vulkan/VulkanPhysicalDevice.hpp>
#include <renderer/vulkan/VulkanDescriptorSet.hpp>

#include <algorithm>

#ifndef ROUND_UP
#define ROUND_UP(v, powerOf2Alignment) (((v) + (powerOf2Alignment)-1) & ~((powerOf2Alignment)-1))
#endif

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




	// Calculate SBTS table size
	VkDeviceSize sbtSize = 0;
	VkDeviceSize rayGenEntrySize = GetEntrySize(m_rayGen);
	VkDeviceSize missEntrySize = GetEntrySize(m_miss);
	VkDeviceSize hitGroupEntrySize = GetEntrySize(m_hitGroup);


	sbtSize = rayGenEntrySize * static_cast<VkDeviceSize>(m_rayGen.size())
		+ missEntrySize * static_cast<VkDeviceSize>(m_miss.size())
		+ hitGroupEntrySize * static_cast<VkDeviceSize>(m_hitGroup.size());


	VulkanCommon::CreateBuffer(
		m_device,
		sbtSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
		m_shaderBindingTable
	);


	{
		VkDeviceSize progIdSize = m_device->GetVulkanPhysicalDevice()->GetPhysicalDeviceRayTracingProperties()->shaderGroupHandleSize;

		uint32_t groupCount = static_cast<uint32_t>(m_rayGen.size())
			+ static_cast<uint32_t>(m_miss.size())
			+ static_cast<uint32_t>(m_hitGroup.size());

		auto shaderHandleStorage = new uint8_t[groupCount * progIdSize];
		VkResult code = vkGetRayTracingShaderGroupHandlesNV(*m_device->GetVulkanDevice(), m_pipeline, 0, groupCount,
				progIdSize * groupCount, shaderHandleStorage);


		VulkanCommon::MapBufferMemory(m_device, m_shaderBindingTable, m_shaderBindingTable.size);


		uint8_t* data = static_cast<uint8_t*>(m_shaderBindingTable.mapped_memory);


		VkDeviceSize offset = 0;

		offset = CopyShaderData(data, m_rayGen, rayGenEntrySize, shaderHandleStorage);
		data += offset;

		offset = CopyShaderData(data, m_miss, missEntrySize, shaderHandleStorage);
		data += offset;

		offset = CopyShaderData(data, m_hitGroup, hitGroupEntrySize, shaderHandleStorage);



		VulkanCommon::UnMapBufferMemory(m_device, m_shaderBindingTable);


	}





	return true;
}

void Renderer::Vulkan::VulkanRaytracePipeline::DestroyPipeline()
{
}

void Renderer::Vulkan::VulkanRaytracePipeline::AttachToCommandBuffer(VkCommandBuffer & command_buffer)
{
	vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, m_pipeline);


	for (auto it = m_descriptor_sets.begin(); it != m_descriptor_sets.end(); it++)
	{
		vkCmdBindDescriptorSets(
			command_buffer,
			VK_PIPELINE_BIND_POINT_RAY_TRACING_NV,
			m_pipeline_layout,
			it->first,
			1,
			&it->second->GetDescriptorSet(),
			0,
			NULL
		);
	}

	VkDeviceSize rayGenOffset = 0;
	VkDeviceSize missOffset = 16;
	VkDeviceSize missStride = 16;
	VkDeviceSize hitGroupOffset = 48;
	VkDeviceSize hitGroupStride = 16;

	vkCmdTraceRaysNV(command_buffer, m_shaderBindingTable.buffer, rayGenOffset,
		m_shaderBindingTable.buffer, missOffset, missStride,
		m_shaderBindingTable.buffer, hitGroupOffset, hitGroupStride,
		VK_NULL_HANDLE, 0, 0, 1080,
		720, 1);
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

void Renderer::Vulkan::VulkanRaytracePipeline::AddRayGenerationProgram(uint32_t group, const std::vector<unsigned char>& inlineData)
{
	m_rayGen.push_back(SBTEntry(group, inlineData));
}

void Renderer::Vulkan::VulkanRaytracePipeline::AddMissProgram(uint32_t group, const std::vector<unsigned char>& inlineData)
{
	m_miss.push_back(SBTEntry(group, inlineData));
}

void Renderer::Vulkan::VulkanRaytracePipeline::AddHitGroup(uint32_t group, const std::vector<unsigned char>& inlineData)
{
	m_hitGroup.push_back(SBTEntry(group, inlineData));
}

VkDeviceSize Renderer::Vulkan::VulkanRaytracePipeline::GetEntrySize(std::vector<SBTEntry> entries)
{
	VkDeviceSize progIdSize = m_device->GetVulkanPhysicalDevice()->GetPhysicalDeviceRayTracingProperties()->shaderGroupHandleSize;
	size_t maxArgs = 0;
	for (const auto& shader : entries)
	{
		maxArgs = maxArgs < shader.m_inlineData.size() ? shader.m_inlineData.size() : maxArgs;
	}
	// A SBT entry is made of a program ID and a set of 4-byte parameters (offsets or push constants)
	VkDeviceSize entrySize = progIdSize + static_cast<VkDeviceSize>(maxArgs);

	// The entries of the shader binding table must be 16-bytes-aligned
	entrySize = ROUND_UP(entrySize, 16);

	return entrySize;
}

VkDeviceSize Renderer::Vulkan::VulkanRaytracePipeline::CopyShaderData(uint8_t * outputData, const std::vector<SBTEntry>& shaders, VkDeviceSize entrySize, const uint8_t * shaderHandleStorage)
{
	VkDeviceSize progIdSize = m_device->GetVulkanPhysicalDevice()->GetPhysicalDeviceRayTracingProperties()->shaderGroupHandleSize;
	uint8_t* pData = outputData;
	for (const auto& shader : shaders)
	{
		// Copy the shader identifier that was previously obtained with
		// vkGetRayTracingShaderGroupHandlesNV
		memcpy(pData, shaderHandleStorage + shader.m_groupIndex * progIdSize, progIdSize);

		// Copy all its resources pointers or values in bulk
		if (!shader.m_inlineData.empty())
		{
			memcpy(pData + progIdSize, shader.m_inlineData.data(), shader.m_inlineData.size());
		}

		pData += entrySize;
	}
	// Return the number of bytes actually written to the output buffer
	return static_cast<uint32_t>(shaders.size()) * entrySize;
}


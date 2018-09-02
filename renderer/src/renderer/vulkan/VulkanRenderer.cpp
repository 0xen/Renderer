#include <renderer/vulkan/VulkanRenderer.hpp>
#include <renderer/vulkan/VulkanInstance.hpp>
#include <renderer/vulkan/VulkanPhysicalDevice.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanSwapchain.hpp>
#include <renderer\vulkan\VulkanUniformBuffer.hpp>
#include <renderer\vulkan\VulkanVertexBuffer.hpp>
#include <renderer\vulkan\VulkanIndexBuffer.hpp>
#include <renderer\vulkan\VulkanComputePipeline.hpp>
#include <renderer\vulkan\VulkanGraphicsPipeline.hpp>
#include <renderer\vulkan\VulkanComputeProgram.hpp>
#include <renderer\vulkan\VulkanModelPool.hpp>
#include <renderer\vulkan\VulkanTextureBuffer.hpp>
#include <renderer\vulkan\VulkanDescriptor.hpp>
#include <renderer\vulkan\VulkanDescriptorPool.hpp>
#include <renderer\vulkan\VulkanDescriptorSet.hpp>


#include <assert.h>

using namespace Renderer;
using namespace Renderer::Vulkan;
VulkanRenderer::VulkanRenderer() : IRenderer()
{
}

VulkanRenderer::~VulkanRenderer()
{
	Stop();
}

bool VulkanRenderer::Start(Renderer::NativeWindowHandle* window_handle)
{
	m_window_handle = window_handle;

	m_instance = new VulkanInstance();
	Status::ErrorCheck(m_instance);
	if (HasError())return false;

	CreateSurface(window_handle);

	m_physical_device = VulkanPhysicalDevice::GetPhysicalDevice(m_instance, m_surface);
	Status::ErrorCheck(m_physical_device);
	if (HasError())return false;

	m_device = new VulkanDevice(m_instance, m_physical_device);
	Status::ErrorCheck(m_device);
	if (HasError())return false;

	m_swapchain = new VulkanSwapchain(m_instance, m_device, &m_surface, window_handle);
	Status::ErrorCheck(m_swapchain);
	if (HasError())return false;


	return true;
}

void Renderer::Vulkan::VulkanRenderer::InitilizeImGUI()
{
	/*ImGui::StyleColorsDark();

	// Color scheme
	ImGuiStyle& style = ImGui::GetStyle();
	style.Colors[ImGuiCol_TitleBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.6f);
	style.Colors[ImGuiCol_TitleBgActive] = ImVec4(1.0f, 0.0f, 0.0f, 0.8f);
	style.Colors[ImGuiCol_MenuBarBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
	style.Colors[ImGuiCol_Header] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
	style.Colors[ImGuiCol_CheckMark] = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);*/


	ImGuiIO& io = ImGui::GetIO();
	io.DisplaySize = ImVec2(m_window_handle->width, m_window_handle->height);
	io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

	m_imgui_pipeline = dynamic_cast<VulkanGraphicsPipeline*>(CreateGraphicsPipeline({
		{ ShaderStage::VERTEX_SHADER, "../../renderer-demo/Shaders/ImGUI/vert.spv" },
		{ ShaderStage::FRAGMENT_SHADER, "../../renderer-demo/Shaders/ImGUI/frag.spv" }
		}));


	m_imgui_pipeline->AttachVertexBinding({
		VertexInputRate::INPUT_RATE_VERTEX,
		{
			{ 0, DataFormat::R32G32_FLOAT,offsetof(ImDrawVert,pos) },
			{ 1, DataFormat::R32G32_FLOAT,offsetof(ImDrawVert,uv) },
			{ 2, DataFormat::R8G8B8A8_UNORM,offsetof(ImDrawVert,col) },
		},
		sizeof(ImDrawVert),
		0
		});



	// Create font texture
	unsigned char* font_data;
	int font_width, font_height;
	//io.Fonts->AddFontFromFileTTF("../../third_party/imgui/misc/fonts/Cousine-Regular.ttf", 16.0f);
	io.Fonts->GetTexDataAsRGBA32(&font_data, &font_width, &font_height);
	m_font_texture = dynamic_cast<VulkanTextureBuffer*>(CreateTextureBuffer(font_data, Renderer::DataFormat::R8G8B8A8_FLOAT, font_width, font_height));

	io.Fonts->TexID = (ImTextureID)(intptr_t)m_font_texture->GetImage();


	font_texture_pool = CreateDescriptorPool({
		CreateDescriptor(Renderer::DescriptorType::IMAGE_SAMPLER, Renderer::ShaderStage::FRAGMENT_SHADER, 0),
		});
	m_imgui_pipeline->AttachDescriptorPool(font_texture_pool);

	texture_descriptor_set = font_texture_pool->CreateDescriptorSet();
	texture_descriptor_set->AttachBuffer(0, m_font_texture);
	texture_descriptor_set->UpdateSet();
	m_imgui_pipeline->AttachDescriptorSet(0, texture_descriptor_set);


	m_imgui_pipeline->Build();


	m_command_buffers.resize(m_swapchain->GetImageCount());
	// Setup the allocation info
	VkCommandBufferAllocateInfo alloc_info = VulkanInitializers::CommandBufferAllocateInfo(
		*m_device->GetGraphicsCommandPool(),
		static_cast<uint32_t>(m_command_buffers.size())
	);
	// Allocate the buffers
	ErrorCheck(vkAllocateCommandBuffers(
		*m_device->GetVulkanDevice(),
		&alloc_info,
		m_command_buffers.data()
	));
}
void Renderer::Vulkan::VulkanRenderer::RenderImGUI()
{

	ImDrawData* imDrawData = ImGui::GetDrawData();

	if (vertexBuffer != nullptr && vertexBuffer->GetElementCount() != imDrawData->TotalVtxCount)
	{
		delete vertexBuffer;
		vertexBuffer = nullptr;
		delete vertexData;
	}
	if (indexBuffer != nullptr && indexBuffer->GetElementCount() != imDrawData->TotalIdxCount)
	{
		delete indexBuffer;
		indexBuffer = nullptr;
		delete indexData;
	}

	if (vertexBuffer == nullptr)
	{
		vertexData = new ImDrawVert[imDrawData->TotalIdxCount];
		vertexBuffer = dynamic_cast<VulkanVertexBuffer*>(CreateVertexBuffer(vertexData, sizeof(ImDrawVert), imDrawData->TotalVtxCount));
	}
	if (indexBuffer == nullptr)
	{
		indexData = new ImDrawIdx[imDrawData->TotalIdxCount];
		indexBuffer = dynamic_cast<VulkanIndexBuffer*>(CreateIndexBuffer(indexData, sizeof(ImDrawIdx), imDrawData->TotalIdxCount));
	}


	// Upload data
	ImDrawVert* vtxDst = vertexData;
	ImDrawIdx* idxDst = indexData;

	for (int n = 0; n < imDrawData->CmdListsCount; n++)
	{
		const ImDrawList* cmd_list = imDrawData->CmdLists[n];
		memcpy(vtxDst, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
		memcpy(idxDst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
		vtxDst += cmd_list->VtxBuffer.Size;
		idxDst += cmd_list->IdxBuffer.Size;
	}
	vertexBuffer->SetData();
	indexBuffer->SetData();




	VkCommandBufferBeginInfo begin_info = VulkanInitializers::CommandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
	VkClearValue clearValues[2];
	clearValues[0].color = { { 0.1f, 0.1f, 0.5f, 1.0f } };
	clearValues[1].depthStencil = { 1.0f, 0 };

	VkRenderPassBeginInfo renderPassBeginInfo = {};
	renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.renderPass = *m_swapchain->GetRenderPass();
	renderPassBeginInfo.renderArea.offset.x = 0;
	renderPassBeginInfo.renderArea.offset.y = 0;
	renderPassBeginInfo.renderArea.extent.width = m_window_handle->width;
	renderPassBeginInfo.renderArea.extent.height = m_window_handle->height;
	renderPassBeginInfo.clearValueCount = 2;
	renderPassBeginInfo.pClearValues = clearValues;

	for (int32_t i = 0; i < m_command_buffers.size(); ++i)
	{

		vkResetCommandBuffer(
			m_command_buffers[i],
			0
		);

		renderPassBeginInfo.framebuffer = m_swapchain->GetSwapchainFrameBuffers()[i];


		vkBeginCommandBuffer(m_command_buffers[i], &begin_info);

		vkCmdBeginRenderPass(m_command_buffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);




		m_imgui_pipeline->AttachToCommandBuffer(m_command_buffers[i]);

		vkCmdSetLineWidth(m_command_buffers[i], 1.0f);

		VkDeviceSize offsets[1] = { 0 };
		vkCmdBindVertexBuffers(m_command_buffers[i], 0, 1, &vertexBuffer->GetBufferData()->buffer, offsets);
		vkCmdBindIndexBuffer(m_command_buffers[i], indexBuffer->GetBufferData()->buffer, 0, VK_INDEX_TYPE_UINT16);

		ImGuiIO& io = ImGui::GetIO();
		const VkViewport viewport = VulkanInitializers::Viewport(io.DisplaySize.x, io.DisplaySize.y, 0, 0, 0.0f, 1.0f);
		vkCmdSetViewport(m_command_buffers[i], 0, 1, &viewport);


		ImDrawData* imDrawData = ImGui::GetDrawData();
		// Render the command lists:
		int vtx_offset = 0;
		int idx_offset = 0;
		for (int n = 0; n < imDrawData->CmdListsCount; n++)
		{
			const ImDrawList* cmd_list = imDrawData->CmdLists[n];
			for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++)
			{
				const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];
				if (pcmd->UserCallback)
				{
					pcmd->UserCallback(cmd_list, pcmd);
				}
				else
				{
					// Apply scissor/clipping rectangle
					// FIXME: We could clamp width/height based on clamped min/max values.
					VkRect2D scissorRect;
					scissorRect.offset.x = std::max((int32_t)(pcmd->ClipRect.x), 0);
					scissorRect.offset.y = std::max((int32_t)(pcmd->ClipRect.y), 0);
					scissorRect.extent.width = (uint32_t)(pcmd->ClipRect.z - pcmd->ClipRect.x);
					scissorRect.extent.height = (uint32_t)(pcmd->ClipRect.w - pcmd->ClipRect.y);
					vkCmdSetScissor(m_command_buffers[i], 0, 1, &scissorRect);


					// Draw
					vkCmdDrawIndexed(m_command_buffers[i], pcmd->ElemCount, 1, idx_offset, vtx_offset, 0);
				}
				idx_offset += pcmd->ElemCount;
			}
			vtx_offset += cmd_list->VtxBuffer.Size;
		}



		/*ImDrawData* imDrawData = ImGui::GetDrawData();
		int32_t vertexOffset = 0;
		int32_t indexOffset = 0;
		for (int32_t i = 0; i < imDrawData->CmdListsCount; i++)
		{
			const ImDrawList* cmd_list = imDrawData->CmdLists[i];
			for (int32_t j = 0; j < cmd_list->CmdBuffer.Size; j++)
			{
			
				const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[j];
				VkRect2D scissorRect;
				scissorRect.offset.x = std::max((int32_t)(pcmd->ClipRect.x), 0);
				scissorRect.offset.y = std::max((int32_t)(pcmd->ClipRect.y), 0);
				scissorRect.extent.width = (uint32_t)(pcmd->ClipRect.z - pcmd->ClipRect.x);
				scissorRect.extent.height = (uint32_t)(pcmd->ClipRect.w - pcmd->ClipRect.y);
				vkCmdSetScissor(m_command_buffers[i], 0, 1, &scissorRect);
				//vkCmdDrawIndexed(m_command_buffers[i], pcmd->ElemCount, 1, indexOffset, vertexOffset, 0);
				indexOffset += pcmd->ElemCount;
			}
			vertexOffset += cmd_list->VtxBuffer.Size;
		}*/


		vkCmdEndRenderPass(m_command_buffers[i]);

		vkEndCommandBuffer(m_command_buffers[i]);
	}


}

void VulkanRenderer::Update()
{
	unsigned int currentBuffer = m_swapchain->GetCurrentBuffer();

	VkSubmitInfo sumbit_info = m_swapchain->GetSubmitInfo();
	//m_swapchain->SubmitQueue(currentBuffer, sumbit_info);

	// Submit ImGUI
	sumbit_info.pCommandBuffers = &m_command_buffers[currentBuffer];

	ErrorCheck(vkQueueSubmit(
		*m_device->GetGraphicsQueue(),
		1,
		&sumbit_info,
		VK_NULL_HANDLE
	));
	assert(!HasError() && "Queue submission unsuccessful");

	ErrorCheck(vkQueueWaitIdle(*m_device->GetGraphicsQueue()));
	assert(!HasError());
	// End Submit ImGUI

	m_swapchain->Present();
}

void VulkanRenderer::Stop()
{
	delete m_device;
	delete m_physical_device;
	delete m_instance;
}

void Renderer::Vulkan::VulkanRenderer::Rebuild()
{
	m_swapchain->RebuildSwapchain();
}

IUniformBuffer * Renderer::Vulkan::VulkanRenderer::CreateUniformBuffer(void * dataPtr, unsigned int indexSize, unsigned int elementCount)
{
	return new VulkanUniformBuffer(m_device, dataPtr, indexSize, elementCount);
}

IVertexBuffer * Renderer::Vulkan::VulkanRenderer::CreateVertexBuffer(void * dataPtr, unsigned int indexSize, unsigned int elementCount)
{
	return new VulkanVertexBuffer(m_device, dataPtr, indexSize, elementCount);
}

IIndexBuffer * Renderer::Vulkan::VulkanRenderer::CreateIndexBuffer(void * dataPtr, unsigned int indexSize, unsigned int elementCount)
{
	return new VulkanIndexBuffer(m_device, dataPtr, indexSize, elementCount);
}

IGraphicsPipeline * Renderer::Vulkan::VulkanRenderer::CreateGraphicsPipeline(std::map<ShaderStage, const char*> paths)
{
	VulkanGraphicsPipeline* graphics_pipeline = new VulkanGraphicsPipeline(m_device, m_swapchain, paths);
	m_swapchain->AttachGraphicsPipeline(graphics_pipeline);
	return graphics_pipeline;
}

IComputePipeline * Renderer::Vulkan::VulkanRenderer::CreateComputePipeline(const char * path, unsigned int x, unsigned int y, unsigned int z)
{
	return new VulkanComputePipeline(m_device,path, x, y, z);
}

IComputeProgram * Renderer::Vulkan::VulkanRenderer::CreateComputeProgram()
{
	return new VulkanComputeProgram(m_device);
}

IModelPool * Renderer::Vulkan::VulkanRenderer::CreateModelPool(IVertexBuffer * vertex_buffer, IIndexBuffer * index_buffer)
{
	return new VulkanModelPool(m_device, vertex_buffer, index_buffer);
}

ITextureBuffer * Renderer::Vulkan::VulkanRenderer::CreateTextureBuffer(void * dataPtr, DataFormat format, unsigned int width, unsigned int height)
{
	return new VulkanTextureBuffer(m_device, dataPtr, format, width, height);
}

IDescriptor * Renderer::Vulkan::VulkanRenderer::CreateDescriptor(DescriptorType descriptor_type, ShaderStage shader_stage, unsigned int binding)
{
	return new VulkanDescriptor(descriptor_type, shader_stage, binding);
}

IDescriptorPool * Renderer::Vulkan::VulkanRenderer::CreateDescriptorPool(std::vector<IDescriptor*> descriptors)
{
	return new VulkanDescriptorPool(m_device, descriptors);
}

VkDescriptorType Renderer::Vulkan::VulkanRenderer::ToDescriptorType(DescriptorType descriptor_type)
{
	switch (descriptor_type)
	{
	case UNIFORM:
		return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	case IMAGE_SAMPLER:
		return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	}
	return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

VkShaderStageFlagBits Renderer::Vulkan::VulkanRenderer::ToVulkanShader(ShaderStage stage)
{
	switch(stage)
	{
		case VERTEX_SHADER:
			return VK_SHADER_STAGE_VERTEX_BIT;
		case FRAGMENT_SHADER:
			return VK_SHADER_STAGE_FRAGMENT_BIT;
		case COMPUTE_SHADER:
			return VK_SHADER_STAGE_COMPUTE_BIT;
		case GEOMETRY_SHADER:
			return VK_SHADER_STAGE_GEOMETRY_BIT;
	}
	return VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM;
}

void Renderer::Vulkan::VulkanRenderer::CreateSurface(Renderer::NativeWindowHandle* window_handle)
{
	auto CreateWin32SurfaceKHR = (PFN_vkCreateWin32SurfaceKHR)vkGetInstanceProcAddr(*m_instance->GetInstance(), "vkCreateWin32SurfaceKHR");

	VkWin32SurfaceCreateInfoKHR createInfo = VulkanInitializers::SurfaceCreateInfo(window_handle);

	if (!CreateWin32SurfaceKHR || CreateWin32SurfaceKHR(*m_instance->GetInstance(), &createInfo, nullptr, &m_surface) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create window surface!");
	}
}

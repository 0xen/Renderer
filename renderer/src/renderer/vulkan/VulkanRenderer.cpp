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
#include <renderer\vulkan\VulkanCommon.hpp>

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

	m_running = true;
	return m_running;
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
	if (!built_imgui)
	{
		built_imgui = true;
		ImGuiIO& io = ImGui::GetIO();
		io.DisplaySize = ImVec2(m_window_handle->width, m_window_handle->height);
		io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

		m_imgui_pipeline = dynamic_cast<VulkanGraphicsPipeline*>(CreateGraphicsPipeline({
			{ ShaderStage::VERTEX_SHADER, "../../ImGUI/vert.spv" },
			{ ShaderStage::FRAGMENT_SHADER, "../../ImGUI/frag.spv" }
			}));

		m_imgui_pipeline->UseDepth(false);

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
		m_screen_dim = glm::vec2(1080, 720);
		m_screen_res_buffer = CreateUniformBuffer(&m_screen_dim, BufferChain::Single, sizeof(glm::vec2), 1, true);
		m_screen_res_buffer->SetData(BufferSlot::Primary);
		

		m_screen_res_pool = CreateDescriptorPool({
			CreateDescriptor(Renderer::DescriptorType::UNIFORM, Renderer::ShaderStage::VERTEX_SHADER, 0),
			});
		m_imgui_pipeline->AttachDescriptorPool(m_screen_res_pool);

		m_screen_res_set = m_screen_res_pool->CreateDescriptorSet();
		m_screen_res_set->AttachBuffer(0, m_screen_res_buffer);
		m_screen_res_set->UpdateSet();

		m_imgui_pipeline->AttachDescriptorSet(0, m_screen_res_set);



		// Create font texture
		unsigned char* font_data;
		int font_width, font_height;
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
		m_imgui_pipeline->AttachDescriptorSet(1, texture_descriptor_set);


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


		std::vector<VkAttachmentDescription> attachments = {
			VulkanInitializers::AttachmentDescription(m_swapchain->GetSwapChainImageFormat(), VK_ATTACHMENT_STORE_OP_STORE,VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),	//Color
			VulkanInitializers::AttachmentDescription(VulkanCommon::GetDepthImageFormat(m_device), VK_ATTACHMENT_STORE_OP_DONT_CARE,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)		// Depth
		};
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;

		VkAttachmentReference color_attachment_refrence = VulkanInitializers::AttachmentReference(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0);
		VkAttachmentReference depth_attachment_refrence = VulkanInitializers::AttachmentReference(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);

		VkSubpassDescription subpass = VulkanInitializers::SubpassDescription(color_attachment_refrence, depth_attachment_refrence);

		VkSubpassDependency subpass_dependency = VulkanInitializers::SubpassDependency();

		VkRenderPassCreateInfo render_pass_info = VulkanInitializers::RenderPassCreateInfo(attachments, subpass, subpass_dependency);


		ErrorCheck(vkCreateRenderPass(
			*m_device->GetVulkanDevice(),
			&render_pass_info,
			nullptr,
			&m_render_pass
		));
		assert(!HasError() && "Unable to initialize render pass");

		VkSemaphoreCreateInfo semaphore_info = VulkanInitializers::SemaphoreCreateInfo();

		ErrorCheck(vkCreateSemaphore(*m_device->GetVulkanDevice(), &semaphore_info, nullptr, &m_render_finished_semaphore));
		ErrorCheck(vkCreateSemaphore(*m_device->GetVulkanDevice(), &semaphore_info, nullptr, &m_image_available_semaphore));
	}
}
void Renderer::Vulkan::VulkanRenderer::RenderImGUI()
{

	if (built_imgui)
	{
		ImDrawData* imDrawData = ImGui::GetDrawData();

		if (vertexBuffer != nullptr && vertexBuffer->GetElementCount(BufferSlot::Primary) != imDrawData->TotalVtxCount)
		{
			delete vertexBuffer;
			vertexBuffer = nullptr;
			delete vertexData;
		}
		if (indexBuffer != nullptr && indexBuffer->GetElementCount(BufferSlot::Primary) != imDrawData->TotalIdxCount)
		{
			delete indexBuffer;
			indexBuffer = nullptr;
			delete indexData;
		}

		if (imDrawData->TotalIdxCount != 0 && imDrawData->TotalIdxCount != 0)
		{
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
			vertexBuffer->SetData(BufferSlot::Primary);
			indexBuffer->SetData(BufferSlot::Primary);
		}



		VkCommandBufferBeginInfo begin_info = VulkanInitializers::CommandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
		VkClearValue clearValues[2];
		clearValues[1].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

		VkRenderPassBeginInfo renderPassBeginInfo = {};
		renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassBeginInfo.renderPass = m_render_pass;
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

			if (imDrawData->TotalIdxCount != 0 && imDrawData->TotalIdxCount != 0)
			{
				vkCmdSetLineWidth(m_command_buffers[i], 1.0f);

				VkDeviceSize offsets[1] = { 0 };
				vkCmdBindVertexBuffers(m_command_buffers[i], 0, 1, &vertexBuffer->GetBufferData(BufferSlot::Primary)->buffer, offsets);
				vkCmdBindIndexBuffer(m_command_buffers[i], indexBuffer->GetBufferData(BufferSlot::Primary)->buffer, 0, VK_INDEX_TYPE_UINT16);

				ImGuiIO& io = ImGui::GetIO();
				const VkViewport viewport = VulkanInitializers::Viewport(io.DisplaySize.x, io.DisplaySize.y, 0, 0, 0.0f, 1.0f);
				vkCmdSetViewport(m_command_buffers[i], 0, 1, &viewport);


				// Render the command lists:
				int vtx_offset = 0;
				int idx_offset = 0;
				for (int j = 0; j < imDrawData->CmdListsCount; j++)
				{
					const ImDrawList* cmd_list = imDrawData->CmdLists[j];
					for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++)
					{
						const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];

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

						idx_offset += pcmd->ElemCount;
					}
					vtx_offset += cmd_list->VtxBuffer.Size;
				}
			}

			vkCmdEndRenderPass(m_command_buffers[i]);

			vkEndCommandBuffer(m_command_buffers[i]);
		}
	}
}

void VulkanRenderer::Update()
{
	if (!m_running)return;
	unsigned int currentBuffer = m_swapchain->GetCurrentBuffer();

	m_swapchain->SubmitQueue(currentBuffer);

	std::vector<VkSemaphore> signal_semaphores{
		m_swapchain->GetRenderFinishedSemaphore()
	};

	if (built_imgui)
	{
		signal_semaphores.push_back(m_render_finished_semaphore);
		
		// Submit ImGUI
		VkSubmitInfo sumbit_info = {};
		sumbit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		VkPipelineStageFlags m_wait_stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		sumbit_info.pWaitDstStageMask = &m_wait_stages;
		sumbit_info.commandBufferCount = 1;
		sumbit_info.signalSemaphoreCount = 1;
		sumbit_info.pSignalSemaphores = &m_render_finished_semaphore;
		sumbit_info.pCommandBuffers = &m_command_buffers[currentBuffer];


		ErrorCheck(vkQueueSubmit(
			*m_device->GetGraphicsQueue(),
			1,
			&sumbit_info,
			VK_NULL_HANDLE
		));

		assert(!HasError());
		// End Submit ImGUI
	}
	


	m_swapchain->Present(signal_semaphores);
}

void VulkanRenderer::Stop()
{
	delete m_device;
	delete m_physical_device;
	delete m_instance;
	m_running = false;
}

void Renderer::Vulkan::VulkanRenderer::Rebuild()
{
	m_swapchain->RebuildSwapchain();

	if (built_imgui)
	{
		m_screen_dim = glm::vec2(m_window_handle->width, m_window_handle->height);
		m_screen_res_buffer->SetData(BufferSlot::Primary);
	}
}

IUniformBuffer * Renderer::Vulkan::VulkanRenderer::CreateUniformBuffer(void * dataPtr, BufferChain level, unsigned int indexSize, unsigned int elementCount, bool modifiable)
{
	return new VulkanUniformBuffer(m_device, level, dataPtr, indexSize, elementCount, modifiable);
}

IVertexBuffer * Renderer::Vulkan::VulkanRenderer::CreateVertexBuffer(void * dataPtr, unsigned int indexSize, unsigned int elementCount)
{
	return new VulkanVertexBuffer(m_device,  dataPtr, indexSize, elementCount);
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

IModelPool * Renderer::Vulkan::VulkanRenderer::CreateModelPool(IVertexBuffer * vertex_buffer)
{
	return new VulkanModelPool(m_device, vertex_buffer);
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
	case STORAGE_BUFFER:
		return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
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

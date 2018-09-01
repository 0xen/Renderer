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
	ImGui::StyleColorsDark();

	// Color scheme
	ImGuiStyle& style = ImGui::GetStyle();
	style.Colors[ImGuiCol_TitleBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.6f);
	style.Colors[ImGuiCol_TitleBgActive] = ImVec4(1.0f, 0.0f, 0.0f, 0.8f);
	style.Colors[ImGuiCol_MenuBarBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
	style.Colors[ImGuiCol_Header] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
	style.Colors[ImGuiCol_CheckMark] = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);


	ImGuiIO& io = ImGui::GetIO();
	io.DisplaySize = ImVec2(m_window_handle->width, m_window_handle->height);
	io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);


	// Create font texture
	unsigned char* font_data;
	int font_width, font_height;
	io.Fonts->GetTexDataAsRGBA32(&font_data, &font_width, &font_height);
	m_font_texture = CreateTextureBuffer(font_data, Renderer::DataFormat::R8G8B8A8_FLOAT, font_width, font_height);


	m_imgui_pipeline = CreateGraphicsPipeline({
		{ ShaderStage::VERTEX_SHADER, "../../renderer-demo/Shaders/ui.vert.spv" },
		{ ShaderStage::FRAGMENT_SHADER, "../../renderer-demo/Shaders/ui.frag.spv" }
		});


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


	m_imgui_pipeline->Build();

	
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
		vertexBuffer = CreateVertexBuffer(vertexData, sizeof(ImDrawVert), imDrawData->TotalVtxCount);
	}
	if (indexBuffer == nullptr)
	{
		indexData = new uint16_t[imDrawData->TotalIdxCount];
		indexBuffer = CreateIndexBuffer(indexData, sizeof(uint16_t), imDrawData->TotalIdxCount);
	}










	vertexBuffer->SetData();
	indexBuffer->SetData();
}

void VulkanRenderer::Update()
{
	m_swapchain->Render();
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

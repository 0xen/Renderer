
#include <assert.h>
#include <SDL.h>
#include <SDL_syswm.h>


#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <renderer\vulkan\VulkanSwapchain.hpp>
#include <renderer\vulkan\VulkanDescriptorSet.hpp>
#include <renderer\vulkan\VulkanModelPool.hpp>
#include <renderer\vulkan\VulkanAcceleration.hpp>
#include <renderer\vulkan\VulkanRaytracePipeline.hpp>
#include <renderer/vulkan/VulkanRenderer.hpp>
#include <renderer/vulkan/VulkanTextureBuffer.hpp>
#include <renderer/vulkan/VulkanFlags.hpp>
#include <renderer/vulkan/VulkanVertexBuffer.hpp>
#include <renderer/vulkan/VulkanIndexBuffer.hpp>
#include <renderer/vulkan/VulkanBufferPool.hpp>
#include <renderer/vulkan/VulkanModel.hpp>
#include <renderer/vulkan/VulkanDescriptorPool.hpp>
#include <renderer/vulkan/VulkanRenderPass.hpp>


#include <renderer\VertexBase.hpp>

#include "obj_loader.h"
#include <lodepng.h>

#include <vector>
#include <iostream>
#include <sstream>

using namespace Renderer;
using namespace Renderer::Vulkan;

static const int POSITION_BUFFER = 0;


struct RayCamera
{
	glm::mat4 view;
	glm::mat4 proj;
	// #VKRay
	glm::mat4 viewInverse;
	glm::mat4 projInverse;
};


SDL_Window* window;
NativeWindowHandle* window_handle;
VulkanRenderer* renderer;
VulkanRenderPass* render_pass;
VulkanSwapchain* swapchain;
RayCamera rayCamera;
VulkanDescriptorSet* standardRTConfigSet = nullptr;


VulkanGraphicsPipeline* postProcessTintPipeline1;
VulkanGraphicsPipeline* postProcessTintPipeline2;

class MeshVertex
{
public:
	MeshVertex(glm::vec3 position, glm::vec2 uv, glm::vec3 normal, glm::vec3 color) : position(position), uv(uv), normal(normal), color(color) {}
	glm::vec3 position;
	glm::vec2 uv;
	glm::vec3 normal;
	glm::vec3 color;
};

class PositionVertex
{
public:
	PositionVertex(glm::mat4 pos) : pos(pos) {}
	glm::mat4 pos;
};

void WindowSetup(const char* title, int width, int height)
{
	window = SDL_CreateWindow(
		title,
		SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		width, height,
		SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE
	);
	SDL_ShowWindow(window);

	SDL_SysWMinfo info;
	SDL_VERSION(&info.version);
	bool sucsess = SDL_GetWindowWMInfo(window, &info);
	assert(sucsess && "Error, unable to get window info");

	window_handle = new NativeWindowHandle(info.info.win.window, width, height);
	window_handle->clear_color = { 0.2f,0.2f,0.2f,1.0f };
}

void PollWindow()
{

	// Poll Window
	SDL_Event event;
	bool rebuild = false;
	while (SDL_PollEvent(&event) > 0)
	{
		switch (event.type)
		{
		case SDL_QUIT:
			renderer->Stop();
			break;
		case SDL_WINDOWEVENT:
			switch (event.window.event)
			{
				//Get new dimensions and repaint on window size change
			case SDL_WINDOWEVENT_SIZE_CHANGED:
				window_handle->width = event.window.data1;
				window_handle->height = event.window.data2;

				swapchain->RebuildSwapchain();

				standardRTConfigSet->AttachBuffer(1, { renderer->GetSwapchain()->GetRayTraceStagingBuffer() });
				standardRTConfigSet->UpdateSet();

				render_pass->Rebuild();
				break;
			}
			break;
		}
	}
}
void DestroyWindow()
{
	SDL_DestroyWindow(window);

	delete window_handle;
}


struct Vertex
{
	glm::vec3 pos;
	glm::vec3 nrm;
	glm::vec3 color;
	glm::vec2 texCoord;
	int       matID = 0;

	static auto getBindingDescription();
	static auto getAttributeDescriptions();
};

struct Light
{
	glm::vec3 position;
	float intensity;
	glm::vec3 color;
	float padding;
};

std::vector<VulkanTextureBuffer*> textures;
std::vector<VkDescriptorImageInfo> texture_descriptors;
std::vector<MatrialObj> materials;



unsigned int used_vertex = 0;
unsigned int used_index = 0;

const unsigned int vertex_max = 1000000;
const unsigned int index_max = 1000000;

Vertex all_vertexs[vertex_max];
uint32_t all_indexs[index_max];

VulkanVertexBuffer* vertexBuffer;
VulkanIndexBuffer* indexBuffer;

void LoadTexture(std::string path)
{
	std::vector<unsigned char> image; //the raw pixels
	unsigned width;
	unsigned height;
	unsigned error = lodepng::decode(image, width, height, path);
	if (error)
	{
		std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
		return;
	}
	VulkanTextureBuffer* texture = renderer->CreateTextureBuffer(image.data(), VkFormat::VK_FORMAT_R8G8B8A8_UNORM, width, height);
	texture->SetData(BufferSlot::Primary);
	textures.push_back(texture);
	texture_descriptors.push_back(texture->GetDescriptorImageInfo(BufferSlot::Primary));
}

VulkanModelPool* LoadModel(std::string path)
{
	ObjLoader<Vertex> loader;
	loader.loadModel(path);

	uint32_t m_nbVertices = static_cast<uint32_t>(loader.m_vertices.size());
	uint32_t m_nbIndices = static_cast<uint32_t>(loader.m_indices.size());

	unsigned int vertexStart = used_vertex;
	unsigned int indexStart = used_index;

	for (uint32_t& index : loader.m_indices)
	{
		all_indexs[used_index] = index;
		used_index++;
	}

	for (Vertex& vertex : loader.m_vertices)
	{
		vertex.matID += materials.size();
		all_vertexs[used_vertex] = vertex;
		used_vertex++;
	}



	unsigned int offset = texture_descriptors.size();
	for (auto& material : loader.m_materials)
	{
		if (material.textureID >= 0)
			material.textureID += offset;
		else
			material.textureID = 0; // Set to default texture

		if (material.metalicTextureID >= 0)
			material.metalicTextureID += offset;
		else
			material.metalicTextureID = 0; // Set to default texture

		if (material.roughnessTextureID >= 0)
			material.roughnessTextureID += offset;
		else
			material.roughnessTextureID = 0; // Set to default texture

		materials.push_back(material);
	}

	for (auto& texturePath : loader.m_textures)
	{
		std::stringstream ss;

		ss << "../../renderer-demo/media/scenes/" << texturePath;

		LoadTexture(ss.str());
	}

	return renderer->CreateModelPool(vertexBuffer, vertexStart, m_nbVertices, indexBuffer, indexStart, m_nbIndices, ModelPoolUsage::MultiMesh);
}


int main(int argc, char **argv)
{

	renderer = new VulkanRenderer();

	WindowSetup("Renderer", 1080, 720);

	// If the rendering was not fully created, error out
	assert(renderer != nullptr && "Error, renderer instance could not be created");

	renderer->Start(window_handle, VulkanFlags::Raytrace/* | VulkanFlags::ActiveCMDRebuild*/);


	swapchain = renderer->GetSwapchain();
	render_pass = renderer->CreateRenderPass(1);


	// Ray camera
	glm::mat4 mPos = glm::mat4(1.0f);
	mPos = glm::translate(mPos, glm::vec3(0, 0, 0));
	mPos = glm::scale(mPos, glm::vec3(1.0f, 1.0f, 1.0f));

	rayCamera.view = glm::mat4(1.0f);
	rayCamera.view = glm::scale(rayCamera.view, glm::vec3(1.0f, 1.0f, 1.0f));
	rayCamera.view = glm::translate(rayCamera.view, glm::vec3(0.0f, 0.0f, -1.0f));
	float aspectRatio = ((float)1080) / ((float)720);
	rayCamera.proj = glm::perspective(glm::radians(65.0f), aspectRatio, 0.1f, 1000.0f);


	// Need to flip the projection as GLM was made for OpenGL
	rayCamera.proj[1][1] *= -1;

	// #VKRay
	rayCamera.viewInverse = glm::inverse(rayCamera.view);
	rayCamera.projInverse = glm::inverse(rayCamera.proj);

	// Only flip the Y when the raytrace camera has out instance as it dose not flip
	rayCamera.proj[1][1] *= -1;  // Inverting Y for Vulkan



	vertexBuffer = renderer->CreateVertexBuffer(all_vertexs, sizeof(Vertex), vertex_max);
	indexBuffer = renderer->CreateIndexBuffer(all_indexs, sizeof(uint32_t), index_max);

	// Define a default texture
	LoadTexture("../../renderer-demo/media/scenes/white.png");
	std::vector<Vertex> vertexData = {
		{ glm::vec3(1.0f,1.0f,0.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(1.0f,1.0f,0.0f), glm::vec2(0.0f,0.0f) ,-1 },
	{ glm::vec3(1.0f,-1.0f,0.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(0.0f,1.0f,0.0f), glm::vec2(0.0f,1.0f)  ,-1 },
	{ glm::vec3(-1.0f,-1.0f,0.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(.0f,1.0f,1.0f), glm::vec2(1.0f,1.0f)  ,-1 },
	{ glm::vec3(-1.0f,1.0f,0.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(1.0f,0.0f,1.0f), glm::vec2(1.0f,0.0f)  ,-1 },


	};

	std::vector<uint32_t> indexData{
		0,1,2,
		0,2,3,
	};

	



	unsigned int vertexStart = used_vertex;
	unsigned int indexStart = used_index;

	for (uint32_t& index : indexData)
	{
		all_indexs[used_index] = index;
		used_index++;
	}

	for (Vertex& vertex : vertexData)
	{
		vertex.matID += materials.size();
		all_vertexs[used_vertex] = vertex;
		used_vertex++;
	}
	VulkanModelPool* model_pool2PP = renderer->CreateModelPool(vertexBuffer, vertexStart, 0, indexBuffer, indexStart, indexData.size(), ModelPoolUsage::MultiMesh);

	// PP Model
	{
		VulkanModel* model = model_pool2PP->CreateModel();
	}

	VulkanModelPool* sphere_pool = LoadModel("../../renderer-demo/media/scenes/Medieval_building.obj");


	//VulkanModelPool* sphere_pool = LoadModel("../../renderer-demo/media/scenes/CubePBR/cube.obj");
	//VulkanModelPool* sphere_pool = LoadModel("../../renderer-demo/media/scenes/SpherePBR/sphere.obj");





	vertexBuffer->SetData(BufferSlot::Primary);
	indexBuffer->SetData(BufferSlot::Primary);




	glm::mat4* model_position_array = new glm::mat4[1000];
	VulkanUniformBuffer* model_position_buffer1 = renderer->CreateUniformBuffer(model_position_array, BufferChain::Double, sizeof(glm::mat4), 1000, true);

	VulkanBufferPool* position_buffer_pool = new VulkanBufferPool(model_position_buffer1);

	sphere_pool->AttachBufferPool(POSITION_BUFFER, position_buffer_pool);


	VulkanModel* cube = sphere_pool->CreateModel();


	
	glm::mat4 modelPosition = glm::mat4(1.0f);
	modelPosition = glm::translate(modelPosition, glm::vec3(0, 0, 0));
	float scale = 0.2;
	modelPosition = glm::scale(modelPosition, glm::vec3(scale, scale, scale));
	scale = 0.25f;

	cube->SetData(POSITION_BUFFER, modelPosition);
	



	model_position_buffer1->SetData(BufferSlot::Secondery);
	model_position_buffer1->Transfer(BufferSlot::Primary, BufferSlot::Secondery);








	//RayCamera
	VulkanUniformBuffer* cameraInfo = renderer->CreateUniformBuffer(&rayCamera, BufferChain::Single, sizeof(RayCamera), 1, true);
	cameraInfo->SetData(BufferSlot::Primary);

	VertexBase vertex_binding_vertex = {
		VkVertexInputRate::VK_VERTEX_INPUT_RATE_VERTEX,
		{
			{ 0, VkFormat::VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,pos) },
			{ 1, VkFormat::VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,nrm) },
			{ 2, VkFormat::VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,color) },
			{ 3, VkFormat::VK_FORMAT_R32G32_SFLOAT,offsetof(Vertex,texCoord) },
			{ 4, VkFormat::VK_FORMAT_R32_UINT,offsetof(Vertex,matID) },
		},
		sizeof(Vertex),
		0
	};

	//{
	//	postProcessTintPipeline1 = renderer->CreateGraphicsPipeline(render_pass, {
	//		{ VkShaderStageFlagBits::VK_SHADER_STAGE_VERTEX_BIT, "../../renderer-demo/Shaders/PP/Tint/vert.spv" },
	//		{ VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT, "../../renderer-demo/Shaders/PP/Tint/frag.spv" }
	//		});


	//	// Config base pipeline
	//	{
	//		VulkanGraphicsPipelineConfig& config = postProcessTintPipeline1->GetGraphicsPipelineConfig();
	//		config.allow_darivatives = true;
	//		config.culling = VkCullModeFlagBits::VK_CULL_MODE_NONE;
	//		config.subpass = 1;
	//		config.input = ATTACHMENT;
	//		config.use_depth_stencil = false;
	//	}

	//	// Define the layout of the input coming to the pipeline from the swapchain
	//	postProcessTintPipeline1->AttachDescriptorPool(render_pass->GetInputAttachmentsReadPool());

	//	postProcessTintPipeline1->AttachVertexBinding(vertex_binding_vertex);/*
	//																		postProcessTintPipeline->AttachDescriptorPool(texture_pool);
	//																		postProcessTintPipeline->AttachDescriptorSet(0, texture_descriptor_set1);*/
	//	postProcessTintPipeline1->Build();
	//}


	//postProcessTintPipeline1->AttachModelPool(model_pool2PP);
	//render_pass->AttachGraphicsPipeline(postProcessTintPipeline1);

	//{
	//	postProcessTintPipeline2 = renderer->CreateGraphicsPipeline(render_pass, {
	//		{ VkShaderStageFlagBits::VK_SHADER_STAGE_VERTEX_BIT, "../../renderer-demo/Shaders/PP/Tint2/vert.spv" },
	//		{ VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT, "../../renderer-demo/Shaders/PP/Tint2/frag.spv" }
	//		});


	//	// Config base pipeline
	//	{
	//		VulkanGraphicsPipelineConfig& config = postProcessTintPipeline2->GetGraphicsPipelineConfig();
	//		config.allow_darivatives = true;
	//		config.culling = VkCullModeFlagBits::VK_CULL_MODE_NONE;
	//		config.subpass = 2;
	//		config.input = COMBINED_IMAGE_SAMPLER;
	//		config.use_depth_stencil = false;
	//	}

	//	// Define the layout of the input coming to the pipeline from the swapchain
	//  postProcessTintPipeline2->AttachDescriptorPool(render_pass->GetCombinedImageSamplerReadPool());

	//	postProcessTintPipeline2->AttachVertexBinding(vertex_binding_vertex);/*
	//																		postProcessTintPipeline->AttachDescriptorPool(texture_pool);
	//																		postProcessTintPipeline->AttachDescriptorSet(0, texture_descriptor_set1);*/
	//	postProcessTintPipeline2->Build();
	//}


	//postProcessTintPipeline2->AttachModelPool(model_pool2PP);
	//render_pass->AttachGraphicsPipeline(postProcessTintPipeline2);


	VulkanRaytracePipeline* ray_pipeline = renderer->CreateRaytracePipeline(render_pass,
		{
			{ VkShaderStageFlagBits::VK_SHADER_STAGE_RAYGEN_BIT_NV,		"../../renderer-demo/Shaders/Raytrace/PBR/Gen/rgen.spv" },
			{ VkShaderStageFlagBits::VK_SHADER_STAGE_MISS_BIT_NV,		"../../renderer-demo/Shaders/Raytrace/PBR/Miss/rmiss.spv" },
			{ VkShaderStageFlagBits::VK_SHADER_STAGE_MISS_BIT_NV,		"../../renderer-demo/Shaders/Raytrace/PBR/Miss/ShadowMiss/rmiss.spv" },
		},
	{
		{ // Involved 
			{ VkShaderStageFlagBits::VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV, "../../renderer-demo/Shaders/Raytrace/PBR/Hitgroups/rchit.spv" },
		},
		{}, // Fall through hit group for lighting shadows
	});
	
	render_pass->AttachGraphicsPipeline(ray_pipeline);

	int groupID = 0;

	// Ray generation entry point
	{
		ray_pipeline->AddRayGenerationProgram(groupID++, {});
	}

	// Miss Program
	{
		ray_pipeline->AddMissProgram(groupID++, {});
		ray_pipeline->AddMissProgram(groupID++, {});
	}

	unsigned int texturedLightingID;
	{
		unsigned int lastID = groupID;


		// Textured Lighting
		texturedLightingID = groupID - lastID;
		ray_pipeline->AddHitGroup(groupID++, {});

		// Shadow Miss
		ray_pipeline->AddHitGroup(groupID++, {});
	}


	ray_pipeline->SetMaxRecursionDepth(5);
	



	ray_pipeline->AttachVertexBinding(vertex_binding_vertex);


	VulkanAcceleration* acceleration = renderer->CreateAcceleration();
	acceleration->Build();

	acceleration->AttachModelPool(sphere_pool, texturedLightingID);
	acceleration->Build();

	struct ModelOffsets
	{
		uint32_t index;
		uint32_t vertex;
		uint32_t position;
	};


	ModelOffsets* offset_allocation_array = new ModelOffsets[1000];
	VulkanUniformBuffer* offset_allocation_array_buffer = renderer->CreateUniformBuffer(offset_allocation_array, BufferChain::Single, sizeof(ModelOffsets), 1000, true);



	unsigned int index = 0;
	for (auto& mp : acceleration->GetModelPools())
	{
		unsigned int index_offset = mp.model_pool->GetIndexOffset();
		unsigned int vertex_offset = mp.model_pool->GetVertexOffset();
		for (auto& model : mp.model_pool->GetModels())
		{
			offset_allocation_array[index].index = index_offset;
			offset_allocation_array[index].vertex = vertex_offset;
			offset_allocation_array[index].position = mp.model_pool->GetModelBufferOffset(model.second, POSITION_BUFFER);
			index++;
		}
	}

	offset_allocation_array_buffer->SetData(BufferSlot::Primary);



	VulkanUniformBuffer* materialbuffer = renderer->CreateUniformBuffer(materials.data(), BufferChain::Single, sizeof(MatrialObj), materials.size(), true);
	materialbuffer->SetData(BufferSlot::Primary);

	std::vector<Light> lights = {
		{ glm::vec3(60.0f, 200.0f, -20.0f), 32000, glm::vec3(1.0f, 0.9f, 0.8f) },
		{ glm::vec3(8.0f, 1.0f, 0.0f), 20, glm::vec3(1.0f, 1.0f, 0.0f) },
		{ glm::vec3(-10.0f, 5.0f, 0.0f), 20, glm::vec3(0.0f, 1.0f, 1.0f) }
	};

	VulkanUniformBuffer* lightBuffer = renderer->CreateUniformBuffer(lights.data(), BufferChain::Single, sizeof(Light), lights.size(), true);
	lightBuffer->SetData(BufferSlot::Primary);




	{
		VulkanDescriptorPool* standardRTConfigPool = renderer->CreateDescriptorPool({
			renderer->CreateDescriptor(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV, VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV, 0),
			renderer->CreateDescriptor(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_NV, 1),
			renderer->CreateDescriptor(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_NV, 2),
			});
		ray_pipeline->AttachDescriptorPool(standardRTConfigPool);

		standardRTConfigSet = static_cast<VulkanDescriptorSet*>(standardRTConfigPool->CreateDescriptorSet());

		standardRTConfigSet->AttachBuffer(0, acceleration->GetDescriptorAcceleration());
		standardRTConfigSet->AttachBuffer(1, { renderer->GetSwapchain()->GetRayTraceStagingBuffer() });
		standardRTConfigSet->AttachBuffer(2, cameraInfo);
		standardRTConfigSet->UpdateSet();

		ray_pipeline->AttachDescriptorSet(0, standardRTConfigSet);
	}


	{
		VulkanDescriptorPool* RTModelPool = renderer->CreateDescriptorPool({
			renderer->CreateDescriptor(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV, 0),
			renderer->CreateDescriptor(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV, 1),
			renderer->CreateDescriptor(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV, 2),
			renderer->CreateDescriptor(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV, 3,texture_descriptors.size()),
			renderer->CreateDescriptor(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV, 4),
			});
		ray_pipeline->AttachDescriptorPool(RTModelPool);

		VulkanDescriptorSet* RTModelPoolSet = static_cast<VulkanDescriptorSet*>(RTModelPool->CreateDescriptorSet());

		RTModelPoolSet->AttachBuffer(0, vertexBuffer);
		RTModelPoolSet->AttachBuffer(1, indexBuffer);
		RTModelPoolSet->AttachBuffer(2, materialbuffer);
		if (texture_descriptors.size() > 0) RTModelPoolSet->AttachBuffer(3, texture_descriptors);
		RTModelPoolSet->AttachBuffer(4, lightBuffer);


		RTModelPoolSet->UpdateSet();

		ray_pipeline->AttachDescriptorSet(1, RTModelPoolSet);
	}


	{
		VulkanDescriptorPool* RTModelInstancePool = renderer->CreateDescriptorPool({
			renderer->CreateDescriptor(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV, 0),
			renderer->CreateDescriptor(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV, 1),
			});
		ray_pipeline->AttachDescriptorPool(RTModelInstancePool);


		VulkanDescriptorSet* RTModelInstanceSet = static_cast<VulkanDescriptorSet*>(RTModelInstancePool->CreateDescriptorSet());

		RTModelInstanceSet->AttachBuffer(0, model_position_buffer1);
		RTModelInstanceSet->AttachBuffer(1, offset_allocation_array_buffer);


		RTModelInstanceSet->UpdateSet();

		ray_pipeline->AttachDescriptorSet(2, RTModelInstanceSet);
	}

	ray_pipeline->Build();

	render_pass->RebuildCommandBuffers();

	float orbit = 4.0f;

	float rotate = 0.0f;

	while (renderer->IsRunning())
	{
		rotate += 0.001f;

		lights[1].position = glm::vec3(cos(rotate)*orbit,0.0f,sin(rotate)*orbit);
		lightBuffer->SetData(BufferSlot::Primary);

		//cube->SetData(POSITION_BUFFER, glm::translate(modelPosition, glm::vec3(0, sin(rotate)*3, 0)));
		cube->SetData(POSITION_BUFFER, glm::rotate(cube->GetData<glm::mat4>(POSITION_BUFFER), 0.001f, glm::vec3(0, 1, 0)));


		model_position_buffer1->SetData(BufferSlot::Primary);

		acceleration->Update();

		render_pass->Render();

		PollWindow();
	}




	DestroyWindow();


	delete renderer;

	return 0;
}
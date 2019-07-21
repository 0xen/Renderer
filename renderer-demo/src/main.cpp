
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
RayCamera rayCamera;
VulkanDescriptorSet* standardRTConfigSet = nullptr;

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

				renderer->Rebuild();

				standardRTConfigSet->AttachBuffer(1, { renderer->GetSwapchain()->GetRayTraceStagingBuffer() });
				standardRTConfigSet->UpdateSet();

				renderer->GetSwapchain()->RequestRebuildCommandBuffers();

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
	glm::vec4 position;
	glm::vec4 color;
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
		if (material.textureID >= 0) material.textureID += offset;
		materials.push_back(material);
	}

	for (auto& texturePath : loader.m_textures)
	{
		std::stringstream ss;
		ss << "../../renderer-demo/media/scenes/" << texturePath;

		std::vector<unsigned char> image; //the raw pixels
		unsigned width;
		unsigned height;


		unsigned error = lodepng::decode(image, width, height, ss.str());
		if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

		VulkanTextureBuffer* texture = renderer->CreateTextureBuffer(image.data(), VkFormat::VK_FORMAT_R8G8B8A8_UNORM, width, height);
		texture->SetData(BufferSlot::Primary);
		textures.push_back(texture);
		texture_descriptors.push_back(texture->GetDescriptorImageInfo(BufferSlot::Primary));
	}

	return renderer->CreateModelPool(vertexBuffer, vertexStart, m_nbVertices, indexBuffer, indexStart, m_nbIndices);
}


int main(int argc, char **argv)
{

	renderer = new VulkanRenderer();

	WindowSetup("Renderer", 1080, 720);

	// If the rendering was not fully created, error out
	assert(renderer != nullptr && "Error, renderer instance could not be created");

	renderer->Start(window_handle, VulkanFlags::Raytrace/* | VulkanFlags::ActiveCMDRebuild*/);



	// Ray camera
	glm::mat4 mPos = glm::mat4(1.0f);
	mPos = glm::translate(mPos, glm::vec3(0, 0, 0));
	mPos = glm::scale(mPos, glm::vec3(1.0f, 1.0f, 1.0f));

	rayCamera.view = glm::mat4(1.0f);
	rayCamera.view = glm::scale(rayCamera.view, glm::vec3(1.0f, 1.0f, 1.0f));
	rayCamera.view = glm::translate(rayCamera.view, glm::vec3(0.0f, 0.0f, -1.5f));
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

	VulkanModelPool* house_pool = LoadModel("../../renderer-demo/media/scenes/Medieval_building.obj");
	VulkanModelPool* sphere_pool = LoadModel("../../renderer-demo/media/scenes/sphere.obj");




	std::vector<Vertex> vertexData = {
		{ glm::vec3(1.0f,1.0f,0.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(1.0f,1.0f,0.0f), glm::vec2(0.0f,0.0f) ,-1 },
		{ glm::vec3(1.0f,-1.0f,0.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(0.0f,1.0f,0.0f), glm::vec2(0.0f,1.0f)  ,-1 },
		{ glm::vec3(-1.0f,-1.0f,0.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(.0f,1.0f,1.0f), glm::vec2(1.0f,1.0f)  ,-1 },
		{ glm::vec3(-1.0f,1.0f,0.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(1.0f,0.0f,1.0f), glm::vec2(1.0f,0.0f)  ,-1 }
	};

	std::vector<uint32_t> indexData{
		0,1,2,
		0,2,3
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

	VulkanModelPool* model_pool3 = renderer->CreateModelPool(vertexBuffer, vertexStart, vertexData.size(), indexBuffer, indexStart, indexData.size());





	vertexBuffer->SetData(BufferSlot::Primary);
	indexBuffer->SetData(BufferSlot::Primary);




	glm::mat4* model_position_array = new glm::mat4[1000];
	VulkanUniformBuffer* model_position_buffer1 = renderer->CreateUniformBuffer(model_position_array, BufferChain::Double, sizeof(glm::mat4), 1000, true);

	VulkanBufferPool* position_buffer_pool = new VulkanBufferPool(model_position_buffer1);

	house_pool->AttachBufferPool(POSITION_BUFFER, position_buffer_pool);
	sphere_pool->AttachBufferPool(POSITION_BUFFER, position_buffer_pool);
	model_pool3->AttachBufferPool(POSITION_BUFFER, position_buffer_pool);


	VulkanModel* sphere = sphere_pool->CreateModel();

	glm::mat4 modelPosition = glm::mat4(1.0f);
	modelPosition = glm::translate(modelPosition, glm::vec3(0, 0, -2));
	float scale = 0.2;
	modelPosition = glm::scale(modelPosition, glm::vec3(scale, scale, scale));
	scale = 0.25f;

	sphere->SetData(POSITION_BUFFER, modelPosition);


	{
		VulkanModel* model = model_pool3->CreateModel();

		glm::mat4 pos = glm::mat4(1.0f);
		pos = glm::translate(pos, glm::vec3(0, 0, 0));
		float s = 1.0f;
		pos = glm::scale(pos, glm::vec3(s, s, s));

		model->SetData(POSITION_BUFFER, modelPosition);
	}


	VulkanModel* house1;
	VulkanModel* house2;
	VulkanModel* house3;
	{


		house1 = house_pool->CreateModel();

		glm::mat4 modelPos = glm::mat4(1.0f);
		modelPos = glm::mat4(1.0f);
		modelPos = glm::translate(modelPos, glm::vec3(0, 0, -2.5));
		modelPos = glm::scale(modelPos, glm::vec3(scale, scale, scale));

		house1->SetData(POSITION_BUFFER, modelPos);
	}
	{


		house2 = house_pool->CreateModel();

		glm::mat4 modelPos = glm::mat4(1.0f);
		modelPos = glm::mat4(1.0f);
		modelPos = glm::translate(modelPos, glm::vec3(-1, 0, 0));
		modelPos = glm::scale(modelPos, glm::vec3(scale, scale, scale));

		house2->SetData(POSITION_BUFFER, modelPos);
	}
	{


		house3 = house_pool->CreateModel();

		glm::mat4 modelPos = glm::mat4(1.0f);
		modelPos = glm::mat4(1.0f);
		modelPos = glm::translate(modelPos, glm::vec3(1, 0, 0));
		modelPos = glm::scale(modelPos, glm::vec3(scale, scale, scale));

		house3->SetData(POSITION_BUFFER, modelPos);
	}


	model_position_buffer1->SetData(BufferSlot::Secondery);
	model_position_buffer1->Transfer(BufferSlot::Primary, BufferSlot::Secondery);










	//RayCamera
	VulkanUniformBuffer* cameraInfo = renderer->CreateUniformBuffer(&rayCamera, BufferChain::Single, sizeof(RayCamera), 1, true);
	cameraInfo->SetData(BufferSlot::Primary);



	/*
	VulkanGraphicsPipeline* pipeline = renderer->CreateGraphicsPipeline({
		{ VkShaderStageFlagBits::VK_SHADER_STAGE_VERTEX_BIT, "../../renderer-demo/Shaders/vert.spv" },
		{ VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT, "../../renderer-demo/Shaders/frag.spv" }
		});



	pipeline->AttachVertexBinding({
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
		});

	pipeline->AttachVertexBinding({
		VkVertexInputRate::VK_VERTEX_INPUT_RATE_INSTANCE,
		{
			{ 5, VkFormat::VK_FORMAT_R32G32B32A32_SFLOAT,0 }
		},
		sizeof(PositionVertex),
		1
		});



	VulkanDescriptorPool* camera_pool = renderer->CreateDescriptorPool({
		renderer->CreateDescriptor(VkDescriptorType::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VkShaderStageFlagBits::VK_SHADER_STAGE_VERTEX_BIT, 0),
		});
	pipeline->AttachDescriptorPool(camera_pool);

	VulkanDescriptorSet* camera_descriptor_set = camera_pool->CreateDescriptorSet();
	camera_descriptor_set->AttachBuffer(0, cameraInfo);
	camera_descriptor_set->UpdateSet();

	pipeline->AttachDescriptorSet(0, camera_descriptor_set);



	VulkanDescriptorPool* texture_pool = renderer->CreateDescriptorPool({
		renderer->CreateDescriptor(VkDescriptorType::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT, 0),
		});
	pipeline->AttachDescriptorPool(texture_pool);




	pipeline->Build();



	std::vector<unsigned char> image; //the raw pixels
	unsigned width;
	unsigned height;
	unsigned error = lodepng::decode(image, width, height, "../../renderer-demo/Images/cobble.png");
	if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

	VulkanTextureBuffer* texture = renderer->CreateTextureBuffer(image.data(), VkFormat::VK_FORMAT_R8G8B8A8_UNORM, width, height);



	VulkanDescriptorSet* texture_descriptor_set1 = texture_pool->CreateDescriptorSet();
	texture_descriptor_set1->AttachBuffer(0, texture);
	texture_descriptor_set1->UpdateSet();

	model_pool3->AttachDescriptorSet(1, texture_descriptor_set1);




	pipeline->AttachModelPool(model_pool3);
	*/


	VulkanRaytracePipeline* ray_pipeline = renderer->CreateRaytracePipeline(
		{
			{ VkShaderStageFlagBits::VK_SHADER_STAGE_RAYGEN_BIT_NV,		"../../renderer-demo/Shaders/Raytrace/Gen/rgen.spv" },
			{ VkShaderStageFlagBits::VK_SHADER_STAGE_MISS_BIT_NV,		"../../renderer-demo/Shaders/Raytrace/Miss/rmiss.spv" },
			{ VkShaderStageFlagBits::VK_SHADER_STAGE_MISS_BIT_NV,		"../../renderer-demo/Shaders/Raytrace/Miss/ShadowMiss/rmiss.spv" },
		},
	{
		{ // Involved 
			{ VkShaderStageFlagBits::VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV, "../../renderer-demo/Shaders/Raytrace/Hitgroups/Reflective/rchit.spv" },
		},
		{ // Involved 
			{ VkShaderStageFlagBits::VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV, "../../renderer-demo/Shaders/Raytrace/Hitgroups/Textured/rchit.spv" },
		},
		{ // Involved 
			{ VkShaderStageFlagBits::VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV, "../../renderer-demo/Shaders/Raytrace/Hitgroups/TexturedLighting/rchit.spv" },
		},
		{}, // Fall through hit group for lighting shadows
	});
	
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

	unsigned int reflectiveID;
	unsigned int texturedID;
	unsigned int texturedLightingID;
	{
		unsigned int lastID = groupID;

		// Reflective
		reflectiveID = groupID - lastID;
		ray_pipeline->AddHitGroup(groupID++, {});

		// Textured
		texturedID = groupID - lastID;
		ray_pipeline->AddHitGroup(groupID++, {});

		// Textured Lighting
		texturedLightingID = groupID - lastID;
		ray_pipeline->AddHitGroup(groupID++, {});

		// Shadow Miss
		ray_pipeline->AddHitGroup(groupID++, {});
	}


	ray_pipeline->SetMaxRecursionDepth(10);
	



	ray_pipeline->AttachVertexBinding({
		VkVertexInputRate::VK_VERTEX_INPUT_RATE_VERTEX,
		{
			{ 0, VkFormat::VK_FORMAT_R32G32B32_SFLOAT,offsetof(MeshVertex,position) },
			{ 1, VkFormat::VK_FORMAT_R32G32_SFLOAT,offsetof(MeshVertex,uv) },
			{ 2, VkFormat::VK_FORMAT_R32G32B32_SFLOAT,offsetof(MeshVertex,normal) },
			{ 3, VkFormat::VK_FORMAT_R32G32B32_SFLOAT,offsetof(MeshVertex,color) },
		},
		sizeof(MeshVertex),
		0
		});


	VulkanAcceleration* acceleration = renderer->CreateAcceleration();
	acceleration->AttachModelPool(house_pool, texturedLightingID);
	acceleration->AttachModelPool(sphere_pool, reflectiveID);
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
		{ glm::vec4(500, 400, 300,0), glm::vec4(1.0f,1.0f,1.0f,1.0f) },
		{ glm::vec4(-500, 400, 300,0), glm::vec4(1.0f,1.0f,1.0f,1.0f) }
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

	float s = 0.0f;

	while (renderer->IsRunning())
	{
		s += 0.003f;

		modelPosition = glm::translate(modelPosition, glm::vec3(0, 0, sin(s) * 0.014f));

		sphere->SetData(POSITION_BUFFER, modelPosition);



		house2->SetData(POSITION_BUFFER, glm::rotate(house2->GetData<glm::mat4>(POSITION_BUFFER), 0.001f, glm::vec3(0, 1, 0)));



		model_position_buffer1->SetData(BufferSlot::Primary);

		acceleration->Update();

		renderer->Update();

		PollWindow();
	}




	DestroyWindow();


	delete renderer;

	return 0;
}
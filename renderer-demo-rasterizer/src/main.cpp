

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

	renderer->Start(window_handle);



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

	VulkanModelPool* model_pool1 = LoadModel("../../renderer-demo/media/scenes/Medieval_building.obj");
	VulkanModelPool* model_pool2 = LoadModel("../../renderer-demo/media/scenes/sphere.obj");




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

	model_pool1->AttachBufferPool(POSITION_BUFFER, position_buffer_pool);
	model_pool2->AttachBufferPool(POSITION_BUFFER, position_buffer_pool);
	model_pool3->AttachBufferPool(POSITION_BUFFER, position_buffer_pool);


	VulkanModel* model1 = model_pool2->CreateModel();

	glm::mat4 modelPosition = glm::mat4(1.0f);
	modelPosition = glm::translate(modelPosition, glm::vec3(0, 0, -2));
	float scale = 0.2;
	modelPosition = glm::scale(modelPosition, glm::vec3(scale, scale, scale));
	scale = 0.25f;

	model1->SetData(POSITION_BUFFER, modelPosition);


	{
		VulkanModel* model = model_pool3->CreateModel();

		glm::mat4 pos = glm::mat4(1.0f);
		pos = glm::translate(pos, glm::vec3(0, 0, 0));
		float s = 1.0f;
		pos = glm::scale(pos, glm::vec3(s, s, s));

		model->SetData(POSITION_BUFFER, modelPosition);
	}


	VulkanModel* model2;
	VulkanModel* model3;
	VulkanModel* model4;
	{


		model2 = model_pool1->CreateModel();

		glm::mat4 modelPos = glm::mat4(1.0f);
		modelPos = glm::mat4(1.0f);
		modelPos = glm::translate(modelPos, glm::vec3(0, 1, 0));
		modelPos = glm::scale(modelPos, glm::vec3(scale, scale, scale));

		model2->SetData(POSITION_BUFFER, modelPos);
	}
	{


		model3 = model_pool1->CreateModel();

		glm::mat4 modelPos = glm::mat4(1.0f);
		modelPos = glm::mat4(1.0f);
		modelPos = glm::translate(modelPos, glm::vec3(-1, 0, 0));
		modelPos = glm::scale(modelPos, glm::vec3(scale, scale, scale));

		model3->SetData(POSITION_BUFFER, modelPos);
	}
	{


		model4 = model_pool1->CreateModel();

		glm::mat4 modelPos = glm::mat4(1.0f);
		modelPos = glm::mat4(1.0f);
		modelPos = glm::translate(modelPos, glm::vec3(1, 0, 0));
		modelPos = glm::scale(modelPos, glm::vec3(scale, scale, scale));

		model4->SetData(POSITION_BUFFER, modelPos);
	}


	model_position_buffer1->SetData(BufferSlot::Secondery);
	model_position_buffer1->Transfer(BufferSlot::Primary, BufferSlot::Secondery);










	//RayCamera
	VulkanUniformBuffer* cameraInfo = renderer->CreateUniformBuffer(&rayCamera, BufferChain::Single, sizeof(RayCamera), 1, true);
	cameraInfo->SetData(BufferSlot::Primary);




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




	float s = 0.0f;

	while (renderer->IsRunning())
	{
		s += 0.01f;

		modelPosition = glm::translate(modelPosition, glm::vec3(0, 0, sin(s) * 0.01f));

		model1->SetData(POSITION_BUFFER, modelPosition);

		model_position_buffer1->SetData(BufferSlot::Primary);

		renderer->Update();

		PollWindow();
	}




	DestroyWindow();


	delete renderer;

	return 0;
}
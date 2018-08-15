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

#include <renderer\IRenderer.hpp>
#include <renderer\VertexBase.hpp>

#include <iostream>

using namespace Renderer;

SDL_Window* window;
NativeWindowHandle* window_handle;

Uint32 GetWindowFlags(RenderingAPI api)
{
	switch (api)
	{
		case VulkanAPI:
			return SDL_WINDOW_VULKAN;
			break;
	}
	return 0;
}

void WindowSetup(RenderingAPI api, const char* title, int width, int height)
{
	window = SDL_CreateWindow(
		title,
		SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		width, height,
		GetWindowFlags(api) | SDL_WINDOW_RESIZABLE
	);
	SDL_ShowWindow(window);

	SDL_SysWMinfo info;
	SDL_VERSION(&info.version);
	assert(SDL_GetWindowWMInfo(window, &info) && "Error, unable to get window info");

	window_handle = new NativeWindowHandle(info.info.win.window, width, height);
	window_handle->clear_color = { 0.2f,0.2f,0.2f,1.0f };
}

void DestroyWindow()
{
	SDL_DestroyWindow(window);

	delete window_handle;
}


class Vertex : public Renderer::VertexBase
{
public:
	Vertex() {}
	Vertex(glm::vec3 _position, glm::vec2 _uv, glm::vec3 _normal, glm::vec3 _color)
	{
		position = _position;
		uv = _uv;
		normal = _normal;
		color = _color;
	}
	glm::vec3 position;
	glm::vec2 uv;
	glm::vec3 normal;
	glm::vec3 color;
};
std::vector<VertexBinding> Vertex::vertex_bindings = {
	{ 0, DataFormat::R32G32B32_FLOAT,offsetof(Vertex,position)},
	{ 1, DataFormat::R32G32_FLOAT,offsetof(Vertex,uv)},
	{ 2, DataFormat::R32G32B32_FLOAT,offsetof(Vertex,normal) },
	{ 8, DataFormat::R32G32B32_FLOAT,offsetof(Vertex,color) }

};
unsigned int Vertex::size = sizeof(Vertex);


struct Camera
{
	glm::mat4 view;
	glm::mat4 projection;
};


int main(int argc, char **argv)
{

	// Define what rendering api we are wanting to use
	RenderingAPI rendering_api = RenderingAPI::VulkanAPI;


	WindowSetup(rendering_api, "Renderer", 1080, 720);

	// Create a instance of the renderer
	IRenderer* renderer = IRenderer::CreateRenderer(rendering_api);

	// If the rendering was not fully created, error out
	assert(renderer != nullptr && "Error, renderer instance could not be created");

	renderer->Start(window_handle);
	std::vector<Vertex> vertexData = {
		Vertex(glm::vec3(1.0f,1.0f,0.0f), glm::vec2(0.0f,0.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(1.0f,1.0f,0.0f)),
		Vertex(glm::vec3(1.0f,-1.0f,0.0f), glm::vec2(0.0f,0.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(0.0f,1.0f,0.0f)),
		Vertex(glm::vec3(-1.0f,-1.0f,0.0f), glm::vec2(0.0f,0.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(.0f,1.0f,1.0f)),
		Vertex(glm::vec3(-1.0f,1.0f,0.0f), glm::vec2(0.0f,0.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(1.0f,0.0f,1.0f))
	};
	std::vector<uint16_t> indexData{
		0,1,2,
		0,2,3
	};
	Camera camera;

	camera.view = glm::mat4(1.0f);
	camera.view = glm::scale(camera.view, glm::vec3(1.0f,1.0f,1.0f));
	camera.view = glm::translate(camera.view, glm::vec3(0.0f, 0.0f, 15.0f));

	float aspectRatio = ((float)1080) / ((float)720);
	camera.projection = glm::perspective(
		glm::radians(45.0f),
		aspectRatio,
		0.1f,
		200.0f
	);
	// Need to flip the projection as GLM was made for OpenGL
	camera.projection[1][1] *= -1;


	IUniformBuffer* cameraBuffer = renderer->CreateUniformBuffer(&camera, sizeof(Camera), 1, Renderer::DescriptorType::UNIFORM, ShaderStage::VERTEX_SHADER, 0);
	cameraBuffer->SetData();

	Renderer::VertexBase* vertex = new Vertex;

	IGraphicsPipeline* pipeline = renderer->CreateGraphicsPipeline({
		{ ShaderStage::VERTEX_SHADER, "../../renderer-demo/Shaders/vert.spv" },
		{ ShaderStage::FRAGMENT_SHADER, "../../renderer-demo/Shaders/frag.spv" }
		}, vertex);

	pipeline->AttachBuffer(cameraBuffer);
	pipeline->Build();


	IVertexBuffer* vertexBuffer = renderer->CreateVertexBuffer(vertexData.data(), sizeof(Vertex), vertexData.size());
	IIndexBuffer* indexBuffer = renderer->CreateIndexBuffer(indexData.data(), sizeof(uint16_t), indexData.size());

	vertexBuffer->SetData();
	indexBuffer->SetData();

	IModelPool* model_pool = renderer->CreateModelPool(vertexBuffer, indexBuffer);

	glm::mat4* model_position_array = new glm::mat4[1];
	IUniformBuffer* model_position_buffer = renderer->CreateUniformBuffer(model_position_array, sizeof(glm::mat4), 1, Renderer::DescriptorType::UNIFORM, ShaderStage::VERTEX_SHADER, 3);

	model_pool->AttachBuffer(0, model_position_buffer);

	IModel* model = model_pool->CreateModel();

	glm::mat4 modelPos = glm::mat4(1.0f);
	modelPos = glm::translate(modelPos, glm::vec3(0, 0, -20));
	modelPos = glm::scale(modelPos, glm::vec3(1.0f, 1.0f, 1.0f));

	model->SetData(0, modelPos);

	model_position_buffer->SetData();

	pipeline->AttachModelPool(model_pool);




	float rot = 0.1f;
	bool running = true;
	while (running)
	{
		// Update all renderer's via there Update function
		IRenderer::UpdateAll();

		// Poll Window
		SDL_Event event;
		while (SDL_PollEvent(&event) > 0)
		{
			switch (event.type)
			{
			case SDL_QUIT:
				running = false;
				break;
			case SDL_WINDOWEVENT:
				switch (event.window.event)
				{
					//Get new dimensions and repaint on window size change
				case SDL_WINDOWEVENT_SIZE_CHANGED:
					window_handle->width = event.window.data1;
					window_handle->height = event.window.data2;
					renderer->Rebuild();
					break;
				}
				break;
			}
		}
	}

	delete model;

	delete model_position_buffer;
	delete model_position_array;

	delete model_pool;

	//delete pipeline;

	delete indexBuffer;
	delete vertexBuffer;

	renderer->Stop();

	delete renderer;

	DestroyWindow();

    return 0;
}

#include <assert.h>
#include <SDL.h>
#include <SDL_syswm.h>

#include <glm/glm.hpp>

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
	glm::vec3 position;
	glm::vec2 uv;
	glm::vec3 normal;
};
std::vector<VertexBinding> Vertex::vertex_bindings = {
	{ 0, DataFormat::R32G32B32_FLOAT,offsetof(Vertex,position)},
	{ 1, DataFormat::R32G32_FLOAT,offsetof(Vertex,uv)},
	{ 2, DataFormat::R32G32B32_FLOAT,offsetof(Vertex,normal)}
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

	std::vector<glm::vec3> vertexData{
		glm::vec3(1.0f,1.0f,1.0f),
		glm::vec3(1.0f,1.0f,1.0f),
		glm::vec3(1.0f,1.0f,1.0f)
	};
	std::vector<uint16_t> indexData{
		0,1,2
	};
	Camera camera;

	IUniformBuffer* cameraBuffer = renderer->CreateUniformBuffer(&camera, sizeof(Camera), 1, Renderer::DescriptorType::UNIFORM, ShaderStage::VERTEX_SHADER, 0);

	IVertexBuffer* vertexBuffer = renderer->CreateVertexBuffer(vertexData.data(), sizeof(glm::vec3), vertexData.size());
	IIndexBuffer* indexBuffer = renderer->CreateIndexBuffer(indexData.data(), sizeof(uint16_t), indexData.size());

	vertexBuffer->SetData();
	indexBuffer->SetData();

	Renderer::VertexBase* vertex = new Vertex;

	IGraphicsPipeline* pipeline = renderer->CreateGraphicsPipeline({
		{ ShaderStage::VERTEX_SHADER, "../../renderer-demo/Shaders/vert.spv" },
		{ ShaderStage::FRAGMENT_SHADER, "../../renderer-demo/Shaders/frag.spv" }
		}, vertex);

	pipeline->AttachBuffer(cameraBuffer);
	pipeline->Build();


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

	//delete pipeline;

	delete indexBuffer;
	delete vertexBuffer;

	renderer->Stop();

	delete renderer;

	DestroyWindow();

    return 0;
}

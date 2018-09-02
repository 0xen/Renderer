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

#include <lodepng.h>
#include <imgui.h>

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


	std::vector<MeshVertex> vertexData = {
		MeshVertex(glm::vec3(1.0f,1.0f,0.0f), glm::vec2(0.0f,0.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(1.0f,1.0f,0.0f)),
		MeshVertex(glm::vec3(1.0f,-1.0f,0.0f), glm::vec2(0.0f,1.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(0.0f,1.0f,0.0f)),
		MeshVertex(glm::vec3(-1.0f,-1.0f,0.0f), glm::vec2(1.0f,1.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(.0f,1.0f,1.0f)),
		MeshVertex(glm::vec3(-1.0f,1.0f,0.0f), glm::vec2(1.0f,0.0f) , glm::vec3(1.0f,1.0f,1.0f),glm::vec3(1.0f,0.0f,1.0f))
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



	std::vector<unsigned char> image; //the raw pixels
	unsigned width;
	unsigned height;
	unsigned error = lodepng::decode(image, width, height, "../../renderer-demo/Images/cobble.png");
	if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

	ITextureBuffer* texture = renderer->CreateTextureBuffer(image.data(), Renderer::DataFormat::R8G8B8A8_FLOAT, width, height);


	IUniformBuffer* cameraBuffer = renderer->CreateUniformBuffer(&camera, sizeof(Camera), 1);
	cameraBuffer->SetData();



	IGraphicsPipeline* pipeline = renderer->CreateGraphicsPipeline({
		{ ShaderStage::VERTEX_SHADER, "../../renderer-demo/Shaders/vert.spv" },
		{ ShaderStage::FRAGMENT_SHADER, "../../renderer-demo/Shaders/frag.spv" }
		});


	pipeline->AttachVertexBinding({
		VertexInputRate::INPUT_RATE_VERTEX,
		{
			{ 0, DataFormat::R32G32B32_FLOAT,offsetof(MeshVertex,position) },
			{ 1, DataFormat::R32G32_FLOAT,offsetof(MeshVertex,uv) },
			{ 2, DataFormat::R32G32B32_FLOAT,offsetof(MeshVertex,normal) },
			{ 3, DataFormat::R32G32B32_FLOAT,offsetof(MeshVertex,color) },
		},
		sizeof(MeshVertex),
		0
		});

	pipeline->AttachVertexBinding({
		VertexInputRate::INPUT_RATE_INSTANCE,
		{ 
			{ 4, DataFormat::MAT4_FLOAT,offsetof(PositionVertex,pos) } 
		},
		sizeof(PositionVertex),
		1 
		});


	IDescriptorPool* camera_pool = renderer->CreateDescriptorPool({
		renderer->CreateDescriptor(Renderer::DescriptorType::UNIFORM, Renderer::ShaderStage::VERTEX_SHADER, 0),
		});
	pipeline->AttachDescriptorPool(camera_pool);

	IDescriptorSet* camera_descriptor_set = camera_pool->CreateDescriptorSet();
	camera_descriptor_set->AttachBuffer(0, cameraBuffer);
	camera_descriptor_set->UpdateSet();

	pipeline->AttachDescriptorSet(0, camera_descriptor_set);



	IDescriptorPool* texture_pool = renderer->CreateDescriptorPool({
		renderer->CreateDescriptor(Renderer::DescriptorType::IMAGE_SAMPLER, Renderer::ShaderStage::FRAGMENT_SHADER, 0),
		});
	pipeline->AttachDescriptorPool(texture_pool);


	pipeline->Build();



	IVertexBuffer* vertexBuffer = renderer->CreateVertexBuffer(vertexData.data(), sizeof(MeshVertex), vertexData.size());
	IIndexBuffer* indexBuffer = renderer->CreateIndexBuffer(indexData.data(), sizeof(uint16_t), indexData.size());

	vertexBuffer->SetData();
	indexBuffer->SetData();







	IModelPool* model_pool1 = renderer->CreateModelPool(vertexBuffer, indexBuffer);

	IDescriptorSet* texture_descriptor_set1 = texture_pool->CreateDescriptorSet();
	texture_descriptor_set1->AttachBuffer(0, texture);
	texture_descriptor_set1->UpdateSet();

	model_pool1->AttachDescriptorSet(1, texture_descriptor_set1);


	glm::mat4* model_position_array1 = new glm::mat4[1];
	IUniformBuffer* model_position_buffer1 = renderer->CreateUniformBuffer(model_position_array1, sizeof(glm::mat4), 1);

	model_pool1->AttachBuffer(0, model_position_buffer1);

	IModel* model1 = model_pool1->CreateModel();

	glm::mat4 modelPos = glm::mat4(1.0f);
	modelPos = glm::translate(modelPos, glm::vec3(2, 0, -20));
	modelPos = glm::scale(modelPos, glm::vec3(1.0f, 1.0f, 1.0f));

	model1->SetData(0, modelPos);


	model_position_buffer1->SetData();

	pipeline->AttachModelPool(model_pool1);


	renderer->InitilizeImGUI();

	ImGuiIO& io = ImGui::GetIO();
	io.DisplaySize = ImVec2(1080, 720);
	io.MouseDrawCursor = true;
	io.Fonts->AddFontFromFileTTF("DroidSans.ttf", 24);
	

	float rot = 0.01;
	float fpn = 0.01;
	bool running = true;
	while (running)
	{
		io.DeltaTime = rot;

		//modelPos = glm::rotate(modelPos, glm::radians(rot), glm::vec3(1.0f, 0.0f, 0.0f));
		//model2->SetData(0, modelPos);

		model1->SetData(0, glm::rotate(model1->GetData<glm::mat4>(0), glm::radians(rot), glm::vec3(0.0f, 0.0f, 1.0f)));

		model_position_buffer1->SetData();

		ImGui::NewFrame();
		ImGui::Begin("Vulkan Example");
		ImGui::Text("Test");
		ImGui::End();
		ImGui::Render();


		/*ImGui::NewFrame();
		ImGui::SetNextWindowSize(ImVec2(0, 0));
		ImGui::SetNextWindowPos(ImVec2(650, 20));

		ImGui::Begin("Vulkan Example", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

		ImGui::SetWindowSize(ImVec2(200, 150));
		ImGui::SetWindowPos(ImVec2(650, 250));
		static float f = 0.0f;
		ImGui::Text("Debug information");
		ImGui::SliderFloat("float", &f, 0.0f, 1.0f);

		ImGui::End();

		ImGui::SetNextWindowPos(ImVec2(650, 20));
		ImGui::ShowTestWindow();
		ImGui::Render();*/


		renderer->RenderImGUI();



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
					io.DisplaySize = ImVec2(window_handle->width, window_handle->height);
					break;
				}
				break;
			case SDL_MOUSEBUTTONDOWN:
				//do whatever you want to do after a mouse button was pressed,
				// e.g.:
				io.MouseDown[0] = event.button.state == SDL_RELEASED;
				std::cout << (int)event.button.button << std::endl;
				break;
			case SDL_MOUSEBUTTONUP:
				//do whatever you want to do after a mouse button was pressed,
				// e.g.:
				io.MouseDown[0] = event.button.state == SDL_RELEASED;
				std::cout << (int)event.button.button << std::endl;
				break;
			case SDL_MOUSEMOTION:

				io.MousePos = ImVec2(event.motion.x, event.motion.y);
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

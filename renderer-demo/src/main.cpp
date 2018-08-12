#include <assert.h>
#include <SDL.h>
#include <SDL_syswm.h>

#include <renderer\IRenderer.hpp>


using namespace Renderer;

SDL_Window* window;
NativeWindowHandle window_handle;

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

void WindowSetup(RenderingAPI api, const char* title, unsigned int width, unsigned int height)
{
	window = SDL_CreateWindow(
		title,
		SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		width, height,
		GetWindowFlags(api)
	);
	SDL_ShowWindow(window);

	SDL_SysWMinfo info;
	SDL_VERSION(&info.version);
	assert(SDL_GetWindowWMInfo(window, &info) && "Error, unable to get window info");

	window_handle = { info.info.win.window };

}



void DestroyWindow()
{
	SDL_DestroyWindow(window);
}

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


	bool running = true;
	while (running)
	{

		// Update all renderer's via there Update function
		IRenderer::UpdateAll();

		// Poll Window
		SDL_Event event;
		while (SDL_PollEvent(&event) > 0)
		{
			if (event.type == SDL_QUIT)
				running = false;
		}
	}

	renderer->Stop();


	delete renderer;

	DestroyWindow();

    return 0;
}

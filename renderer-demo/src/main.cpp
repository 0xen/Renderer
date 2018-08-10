#include <assert.h>

#include <renderer\IRenderer.hpp>


using namespace Renderer;

int main(int argc, char **argv)
{
	// Define what rendering api we are wanting to use
	RenderingAPI rendering_api = RenderingAPI::VulkanAPI;
	// Create a instance of the renderer
	IRenderer* renderer = IRenderer::CreateRenderer(rendering_api);

	// If the rendering was not fully created, error out
	assert(renderer != nullptr && "Error, renderer instance could not be created");

	renderer->Start();

	bool running = true;
	while (running)
	{
		// Update all renderer's via there Update function
		IRenderer::UpdateAll();
	}

	renderer->Stop();


	delete renderer;

    return 0;
}

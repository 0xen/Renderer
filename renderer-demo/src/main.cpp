#include <assert.h>

#include <renderer\IRenderer.hpp>


using namespace Renderer;

int main(int argc, char **argv)
{
	// Define what rendering api we are wanting to use
	RenderingAPI rendering_api = RenderingAPI::Vulkan;
	// Create a instance of the renderer
	IRenderer* renderer = IRenderer::CreateRenderer(rendering_api);
	// If the rendering was not fully created, error out
	
	assert(renderer != nullptr && "Error, renderer instance could not be created");


	delete renderer;

    return 0;
}

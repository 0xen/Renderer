#include <renderer/IGraphicsPipeline.hpp>

Renderer::IGraphicsPipeline::IGraphicsPipeline(std::vector<std::pair<Renderer::ShaderStage, const char*>> paths) :
	IPipeline(paths)
{
}
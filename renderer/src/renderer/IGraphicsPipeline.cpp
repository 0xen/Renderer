#include <renderer/IGraphicsPipeline.hpp>

Renderer::IGraphicsPipeline::IGraphicsPipeline(std::map<ShaderStage, const char*> paths) :
	IPipeline(paths)
{
}
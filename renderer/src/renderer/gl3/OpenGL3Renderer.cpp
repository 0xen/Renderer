#include <renderer/gl3/OpenGL3Renderer.hpp>

#include <assert.h>

using namespace Renderer;
using namespace Renderer::GL3;

Renderer::GL3::OpenGL3Renderer::OpenGL3Renderer()
{
}

Renderer::GL3::OpenGL3Renderer::~OpenGL3Renderer()
{
}

bool Renderer::GL3::OpenGL3Renderer::Start(Renderer::NativeWindowHandle * window_handle)
{
	return false;
}

void Renderer::GL3::OpenGL3Renderer::Update()
{
}

void Renderer::GL3::OpenGL3Renderer::Stop()
{
}

void Renderer::GL3::OpenGL3Renderer::Rebuild()
{
}

IUniformBuffer * Renderer::GL3::OpenGL3Renderer::CreateUniformBuffer(void * dataPtr, unsigned int indexSize, unsigned int elementCount, ShaderStage shader_stage, unsigned int binding)
{
	return nullptr;
}

IVertexBuffer * Renderer::GL3::OpenGL3Renderer::CreateVertexBuffer(void * dataPtr, unsigned int indexSize, unsigned int elementCount)
{
	return nullptr;
}

IIndexBuffer * Renderer::GL3::OpenGL3Renderer::CreateIndexBuffer(void * dataPtr, unsigned int indexSize, unsigned int elementCount)
{
	return nullptr;
}

IGraphicsPipeline * Renderer::GL3::OpenGL3Renderer::CreateGraphicsPipeline(std::map<ShaderStage, const char*> paths)
{
	return nullptr;
}

IComputePipeline * Renderer::GL3::OpenGL3Renderer::CreateComputePipeline(const char * path, unsigned int x, unsigned int y, unsigned int z)
{
	return nullptr;
}

IComputeProgram * Renderer::GL3::OpenGL3Renderer::CreateComputeProgram()
{
	return nullptr;
}

IModelPool * Renderer::GL3::OpenGL3Renderer::CreateModelPool(IVertexBuffer * vertex_buffer, IIndexBuffer * index_buffer)
{
	return nullptr;
}

ITextureBuffer * Renderer::GL3::OpenGL3Renderer::CreateTextureBuffer(void * dataPtr, DataFormat format, unsigned int width, unsigned int height, unsigned int binding)
{
	return nullptr;
}

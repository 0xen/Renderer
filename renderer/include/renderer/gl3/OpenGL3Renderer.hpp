#pragma once

#include <renderer/IRenderer.hpp>
#include <renderer\DescriptorType.hpp>
#include <renderer\ShaderStage.hpp>

namespace Renderer
{
	namespace GL3
	{
		class OpenGL3Renderer : public IRenderer
		{
		public:
			OpenGL3Renderer();
			~OpenGL3Renderer();
			virtual bool Start(Renderer::NativeWindowHandle* window_handle);
			virtual void Update();
			virtual void Stop();
			virtual void Rebuild();
			virtual IUniformBuffer* CreateUniformBuffer(void* dataPtr, unsigned int indexSize, unsigned int elementCount, ShaderStage shader_stage, unsigned int binding);

			virtual IVertexBuffer* CreateVertexBuffer(void* dataPtr, unsigned int indexSize, unsigned int elementCount);

			virtual IIndexBuffer* CreateIndexBuffer(void* dataPtr, unsigned int indexSize, unsigned int elementCount);

			virtual IGraphicsPipeline* CreateGraphicsPipeline(std::map<ShaderStage, const char*> paths);

			virtual IComputePipeline* CreateComputePipeline(const char* path, unsigned int x, unsigned int y, unsigned int z);

			virtual IComputeProgram* CreateComputeProgram();

			virtual IModelPool* CreateModelPool(IVertexBuffer* vertex_buffer, IIndexBuffer* index_buffer);

			virtual ITextureBuffer* CreateTextureBuffer(void* dataPtr, DataFormat format, unsigned int width, unsigned int height, unsigned int binding);
		private:

		};
	}
}
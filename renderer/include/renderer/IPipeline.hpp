#pragma once

#include <renderer\ShaderStage.hpp>

#include <map>

namespace Renderer
{
	class IUniformBuffer;
	class ITextureBuffer;
	class IPipeline
	{
	public:
		IPipeline();
		IPipeline(std::map<ShaderStage, const char*> paths);
		virtual void AttachBuffer(IUniformBuffer* buffer) = 0;
		virtual void AttachBuffer(ITextureBuffer* buffer) = 0;
		virtual bool Build() = 0;
		std::map<ShaderStage, const char*> GetPaths();
	private:
		std::map<ShaderStage, const char*> m_paths;
	};
}
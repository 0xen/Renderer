#pragma once

namespace Renderer
{
	class IUniformBuffer;
	class IPipeline
	{
	public:
		IPipeline();
		IPipeline(const char* path);
		virtual void AttachBuffer(IUniformBuffer* buffer) = 0;
		virtual bool Build() = 0;
		const char* GetPath();
	private:
		const char* m_path;
	};
}
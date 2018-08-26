#pragma once

#include <renderer\ShaderStage.hpp>

#include <map>

namespace Renderer
{
	class IDescriptorPool;
	class IDescriptorSet;
	class IPipeline
	{
	public:
		IPipeline();
		IPipeline(std::map<ShaderStage, const char*> paths);
		virtual void AttachDescriptorPool(IDescriptorPool* buffer) = 0;
		virtual void AttachDescriptorSet(IDescriptorSet* descriptor_set) = 0;
		virtual bool Build() = 0;
		std::map<ShaderStage, const char*> GetPaths();
	private:
		std::map<ShaderStage, const char*> m_paths;
	};
}
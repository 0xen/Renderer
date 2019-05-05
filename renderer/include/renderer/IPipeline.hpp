#pragma once

#include <renderer\ShaderStage.hpp>

#include <map>
#include <vector>

namespace Renderer
{
	class IDescriptorPool;
	class IDescriptorSet;
	class IPipeline
	{
	public:
		IPipeline();
		IPipeline(std::vector<std::pair<Renderer::ShaderStage, const char*>> paths);
		virtual void AttachDescriptorPool(IDescriptorPool* buffer) = 0;
		virtual void AttachDescriptorSet(unsigned int setID, IDescriptorSet* descriptor_set) = 0;
		virtual bool Build() = 0;
		std::vector<std::pair<Renderer::ShaderStage, const char*>> GetPaths();
	private:
		std::vector<std::pair<Renderer::ShaderStage, const char*>> m_paths;
	};
}
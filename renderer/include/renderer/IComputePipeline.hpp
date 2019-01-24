#pragma once

#include <renderer\IPipeline.hpp>

namespace Renderer
{
	class IComputePipeline : public virtual IPipeline
	{
	public:
		IComputePipeline(const char* path, unsigned int x, unsigned int y, unsigned int z);
		virtual ~IComputePipeline() {};
		unsigned int GetX();
		unsigned int GetY();
		unsigned int GetZ();
		void SetX(unsigned int x);
		void SetY(unsigned int y);
		void SetZ(unsigned int z);
	private:
		unsigned int m_x;
		unsigned int m_y;
		unsigned int m_z;
	};
}
#pragma once

namespace Renderer
{
	enum ShaderStage
	{
		VERTEX_SHADER = 0x001,
		FRAGMENT_SHADER = 0x002,
		COMPUTE_SHADER = 0x004,
		GEOMETRY_SHADER = 0x008,

		// Raytracing
		RAY_GEN = 0x010,
		MISS = 0x020,
		ANY_HIT = 0x040,
		CLOSEST_HIT = 0x080,
		INTERSECTION = 0x100
	};
}
#pragma once

namespace Renderer
{
	enum ShaderStage
	{
		VERTEX_SHADER,
		FRAGMENT_SHADER,
		COMPUTE_SHADER,
		GEOMETRY_SHADER,

		// Raytracing
		RAY_GEN,
		MISS,
		ANY_HIT,
		CLOSEST_HIT,
		INTERSECTION
	};
}
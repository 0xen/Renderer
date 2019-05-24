#pragma once


namespace Renderer
{
	namespace Vulkan
	{
		enum VulkanFlags
		{
			None = 0x0,
			Raytrace = 0x1,
			ActiveCMDRebuild = 0x2 // Rebuild CMD render commmand every frame
		};
	}
}
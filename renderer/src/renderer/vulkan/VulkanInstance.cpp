#include <renderer/vulkan/VulkanInstance.hpp>
#include <renderer\vulkan\VulkanCommon.hpp>

using namespace Renderer::Vulkan;

VulkanInstance::VulkanInstance()
{
	SetupLayersAndExtensions();
	InitVulkanInstance();
}

VulkanInstance::~VulkanInstance()
{
	DeInitVulkanInstance();
}

VkInstance * VulkanInstance::GetInstance()
{
	return &m_instance;
}

void VulkanInstance::SetupLayersAndExtensions()
{
	m_instance_extensions.empty();

	m_instance_extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);


#ifdef _WIN32
	m_instance_extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif __APPLE__
	m_instance_extensions.push_back("VK_MVK_macos_surface");
#elif __linux__

#elif __unix__

#elif defined(_POSIX_VERSION)

#else
#   error "Unknown compiler"
#endif

}

void VulkanInstance::InitVulkanInstance()
{
	VkApplicationInfo app_info = VulkanInitializers::ApplicationInfo(
		"Temporary Name",
		VK_MAKE_VERSION(1, 0, 0),
		m_engine_name,
		m_engine_version,
		m_api_version
	);
	VkInstanceCreateInfo create_info = VulkanInitializers::InstanceCreateInfo(
		app_info,
		m_instance_extensions,
		m_instance_layers
	);
	ErrorCheck(vkCreateInstance(
		&create_info,									// Information to pass to the function
		NULL,											// Memory allocation callback
		&m_instance										// The Vulkan instance to be initialized
	));
}

void VulkanInstance::DeInitVulkanInstance()
{
	vkDestroyInstance(
		m_instance,
		NULL
	);
}
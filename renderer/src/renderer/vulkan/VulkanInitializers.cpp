#include <renderer/vulkan/VulkanInitializers.hpp>

VkApplicationInfo Renderer::VulkanInitializers::ApplicationInfo(const char * app_name, uint32_t app_ver, const char * engine_name, uint32_t engine_ver, uint32_t api_version)
{
	VkApplicationInfo app_info = {};
	app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app_info.pApplicationName = app_name;			// Application name
	app_info.applicationVersion = app_ver;		// Application version
	app_info.pEngineName = engine_name;				// Engine name
	app_info.engineVersion = engine_ver;			// Engine version
	app_info.apiVersion = api_version;		// Required API version
	return app_info;
}

VkInstanceCreateInfo Renderer::VulkanInitializers::InstanceCreateInfo(VkApplicationInfo & app_info, std::vector<const char*>& instance_extensions, std::vector<const char*>& instance_layers)
{
	VkInstanceCreateInfo create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	create_info.pApplicationInfo = &app_info;							// Pointer to the application information created
	create_info.enabledExtensionCount = (uint32_t)instance_extensions.size();		// The amount of extensions we wish to enable
	create_info.ppEnabledExtensionNames = instance_extensions.data();	// The raw data of the extensions to enable
	create_info.enabledLayerCount = (uint32_t)instance_layers.size();				// The amount of layers we wish to enable
	create_info.ppEnabledLayerNames = instance_layers.data();			// The raw data of the layers to enable
	return create_info;
}

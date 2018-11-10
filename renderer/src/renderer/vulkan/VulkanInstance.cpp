#include <renderer/vulkan/VulkanInstance.hpp>
#include <renderer\vulkan\VulkanCommon.hpp>

#include <assert.h>
#include <iostream>

using namespace Renderer::Vulkan;

VulkanInstance::VulkanInstance()
{
	SetupLayersAndExtensions();
	assert(CheckLayersSupport() && "Unsupported Layers");
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
	
	m_instance_layers.push_back("VK_LAYER_LUNARG_standard_validation");

	m_instance_extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
	m_instance_extensions.push_back("VK_EXT_debug_report");

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

VKAPI_ATTR VkBool32 VKAPI_CALL MyDebugReportCallback(
	VkDebugReportFlagsEXT       flags,
	VkDebugReportObjectTypeEXT  objectType,
	uint64_t                    object,
	size_t                      location,
	int32_t                     messageCode,
	const char*                 pLayerPrefix,
	const char*                 pMessage,
	void*                       pUserData)
{
	std::cerr << pMessage << std::endl;
	return VK_FALSE;
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
	/*
	PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT =
		reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>
		(vkGetInstanceProcAddr(m_instance, "vkCreateDebugReportCallbackEXT"));
	PFN_vkDebugReportMessageEXT vkDebugReportMessageEXT =
		reinterpret_cast<PFN_vkDebugReportMessageEXT>
		(vkGetInstanceProcAddr(m_instance, "vkDebugReportMessageEXT"));
	PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT =
		reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>
		(vkGetInstanceProcAddr(m_instance, "vkDestroyDebugReportCallbackEXT"));



	VkDebugReportCallbackCreateInfoEXT callbackCreateInfo;
	callbackCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
	callbackCreateInfo.pNext = nullptr;
	callbackCreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT |
		VK_DEBUG_REPORT_WARNING_BIT_EXT |
		VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
	callbackCreateInfo.pfnCallback = &MyDebugReportCallback;
	callbackCreateInfo.pUserData = nullptr;

	VkDebugReportCallbackEXT callback;
	VkResult result = vkCreateDebugReportCallbackEXT(m_instance, &callbackCreateInfo, nullptr, &callback);*/
}

void VulkanInstance::DeInitVulkanInstance()
{
	vkDestroyInstance(
		m_instance,
		NULL
	);
}

bool Renderer::Vulkan::VulkanInstance::CheckLayersSupport()
{

	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());


	for (const char* layerName : m_instance_layers)
	{
		bool layerFound = false;

		for (const auto& layerProperties : availableLayers)
		{
			if (strcmp(layerName, layerProperties.layerName) == 0)
			{
				layerFound = true;
				break;
			}
		}

		if (!layerFound)
		{
			return false;
		}
	}

	return true;
}

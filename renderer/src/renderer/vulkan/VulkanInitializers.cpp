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

VkWin32SurfaceCreateInfoKHR Renderer::VulkanInitializers::SurfaceCreateInfo(Renderer::NativeWindowHandle window_handle)
{
	VkWin32SurfaceCreateInfoKHR createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
	createInfo.hwnd = window_handle.window;
	createInfo.hinstance = GetModuleHandle(nullptr);
	return createInfo;
}

VkDeviceQueueCreateInfo Renderer::VulkanInitializers::DeviceQueueCreate(uint32_t queue_family_index, float queue_priority)
{
	VkDeviceQueueCreateInfo queue_create_info = {};
	queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queue_create_info.queueFamilyIndex = queue_family_index;
	queue_create_info.queueCount = 1;
	queue_create_info.pQueuePriorities = &queue_priority;
	return queue_create_info;
}

VkDeviceCreateInfo Renderer::VulkanInitializers::DeviceCreateInfo(std::vector<VkDeviceQueueCreateInfo>& queue_create_infos, const std::vector<const char*>& device_extensions, VkPhysicalDeviceFeatures & device_features)
{
	VkDeviceCreateInfo create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	create_info.pQueueCreateInfos = queue_create_infos.data();
	create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
	create_info.pEnabledFeatures = &device_features;
	create_info.enabledExtensionCount = (uint32_t)device_extensions.size();
	create_info.ppEnabledExtensionNames = device_extensions.data();
	return create_info;
}

VkCommandPoolCreateInfo Renderer::VulkanInitializers::CommandPoolCreateInfo(uint32_t queue_family_index, VkCommandPoolCreateFlags flags)
{
	VkCommandPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	pool_info.queueFamilyIndex = queue_family_index;
	pool_info.flags = flags;
	return pool_info;
}

VkCommandBufferAllocateInfo Renderer::VulkanInitializers::CommandBufferAllocateInfo(VkCommandPool pool, uint32_t command_buffer_count)
{
	VkCommandBufferAllocateInfo alloc_info = {};
	alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	alloc_info.commandPool = pool;
	alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	alloc_info.commandBufferCount = command_buffer_count;
	return alloc_info;
}

VkCommandBufferBeginInfo Renderer::VulkanInitializers::CommandBufferBeginInfo(VkCommandBufferUsageFlags flag)
{
	VkCommandBufferBeginInfo begin_info = {};
	begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags = flag;
	begin_info.pInheritanceInfo = nullptr;
	return begin_info;
}

VkSubmitInfo Renderer::VulkanInitializers::SubmitInfo(VkCommandBuffer & buffer)
{
	VkSubmitInfo submit_info = {};
	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &buffer;
	return submit_info;
}

VkSubmitInfo Renderer::VulkanInitializers::SubmitInfo(VkCommandBuffer* buffer, uint32_t count)
{
	VkSubmitInfo submit_info = {};
	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = count;
	submit_info.pCommandBuffers = buffer;
	return submit_info;
}

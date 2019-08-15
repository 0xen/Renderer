#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>

#include <assert.h>

VkApplicationInfo Renderer::Vulkan::VulkanInitializers::ApplicationInfo(const char * app_name, uint32_t app_ver, const char * engine_name, uint32_t engine_ver, uint32_t api_version)
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

VkInstanceCreateInfo Renderer::Vulkan::VulkanInitializers::InstanceCreateInfo(VkApplicationInfo & app_info, std::vector<const char*>& instance_extensions, std::vector<const char*>& instance_layers)
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

VkWin32SurfaceCreateInfoKHR Renderer::Vulkan::VulkanInitializers::SurfaceCreateInfo(Renderer::NativeWindowHandle* window_handle)
{
	VkWin32SurfaceCreateInfoKHR createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
	createInfo.hwnd = window_handle->window;
	createInfo.hinstance = GetModuleHandle(nullptr);
	return createInfo;
}

VkDeviceQueueCreateInfo Renderer::Vulkan::VulkanInitializers::DeviceQueueCreate(uint32_t queue_family_index, const float& queue_priority)
{
	VkDeviceQueueCreateInfo queue_create_info = {};
	queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queue_create_info.queueFamilyIndex = queue_family_index;
	queue_create_info.queueCount = 1;
	queue_create_info.pQueuePriorities = &queue_priority;
	return queue_create_info;
}

VkDeviceCreateInfo Renderer::Vulkan::VulkanInitializers::DeviceCreateInfo(std::vector<VkDeviceQueueCreateInfo>& queue_create_infos, const std::vector<const char*>& device_extensions, VkPhysicalDeviceFeatures & device_features)
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

VkDeviceCreateInfo Renderer::Vulkan::VulkanInitializers::DeviceCreateInfo(std::vector<VkDeviceQueueCreateInfo>& queue_create_infos, const std::vector<const char*>& device_extensions, VkPhysicalDeviceFeatures2 & device_features)
{
	VkDeviceCreateInfo create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	create_info.pQueueCreateInfos = queue_create_infos.data();
	create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
	create_info.pEnabledFeatures = &(device_features.features);
	create_info.enabledExtensionCount = (uint32_t)device_extensions.size();
	create_info.ppEnabledExtensionNames = device_extensions.data();
	return create_info;
}

VkCommandPoolCreateInfo Renderer::Vulkan::VulkanInitializers::CommandPoolCreateInfo(uint32_t queue_family_index, VkCommandPoolCreateFlags flags)
{
	VkCommandPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	pool_info.queueFamilyIndex = queue_family_index;
	pool_info.flags = flags;
	return pool_info;
}

VkCommandBufferAllocateInfo Renderer::Vulkan::VulkanInitializers::CommandBufferAllocateInfo(VkCommandPool pool, uint32_t command_buffer_count)
{
	VkCommandBufferAllocateInfo alloc_info = {};
	alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	alloc_info.commandPool = pool;
	alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	alloc_info.commandBufferCount = command_buffer_count;
	return alloc_info;
}

VkCommandBufferBeginInfo Renderer::Vulkan::VulkanInitializers::CommandBufferBeginInfo(VkCommandBufferUsageFlags flag)
{
	VkCommandBufferBeginInfo begin_info = {};
	begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags = flag;
	begin_info.pInheritanceInfo = nullptr;
	return begin_info;
}

VkSubmitInfo Renderer::Vulkan::VulkanInitializers::SubmitInfo(VkCommandBuffer & buffer)
{
	VkSubmitInfo submit_info = {};
	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &buffer;
	return submit_info;
}

VkSubmitInfo Renderer::Vulkan::VulkanInitializers::SubmitInfo(VkCommandBuffer* buffer, uint32_t count)
{
	VkSubmitInfo submit_info = {};

	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.pNext = nullptr;
	submit_info.waitSemaphoreCount = 0;
	submit_info.pWaitSemaphores = nullptr;
	submit_info.pWaitDstStageMask = nullptr;
	submit_info.commandBufferCount = count;
	submit_info.pCommandBuffers = buffer;
	submit_info.signalSemaphoreCount = 0;
	submit_info.pSignalSemaphores = nullptr;
	return submit_info;
}

VkRenderPassBeginInfo Renderer::Vulkan::VulkanInitializers::RenderPassBeginInfo(VkRenderPass render_pass, VkExtent2D swapchain_extent)
{
	VkRenderPassBeginInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	render_pass_info.renderPass = render_pass;
	render_pass_info.renderArea.offset = { 0, 0 };
	render_pass_info.renderArea.extent = swapchain_extent;
	render_pass_info.clearValueCount = 0;
	render_pass_info.pClearValues = nullptr;
	return render_pass_info;
}

VkRenderPassBeginInfo Renderer::Vulkan::VulkanInitializers::RenderPassBeginInfo(VkRenderPass render_pass, VkExtent2D swapchain_extent, std::vector<VkClearValue>& clear_values)
{
	VkRenderPassBeginInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	render_pass_info.renderPass = render_pass;
	render_pass_info.renderArea.offset = { 0, 0 };
	render_pass_info.renderArea.extent = swapchain_extent;
	render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
	render_pass_info.pClearValues = clear_values.data();
	return render_pass_info;
}

VkSwapchainCreateInfoKHR Renderer::Vulkan::VulkanInitializers::SwapchainCreateInfoKHR(VkSurfaceFormatKHR surface_format, VkExtent2D extent, VkPresentModeKHR present_mode, uint32_t image_count, VkSurfaceKHR surface, Renderer::Vulkan::VulkanQueueFamilyIndices indices, Renderer::Vulkan::VulkanSwapChainSupport swap_chain_support)
{
	VkSwapchainCreateInfoKHR create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	create_info.surface = surface; // The surface we are drawing onto
	create_info.minImageCount = image_count; // Amount of swapchain images
	create_info.imageFormat = surface_format.format; // What format will the images be in
	create_info.imageColorSpace = surface_format.colorSpace; // What colors will the images use
	create_info.imageExtent = extent; // Image width and height
	create_info.imageArrayLayers = 1; // Keep at 1
	create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;// | VK_IMAGE_USAGE_STORAGE_BIT; // VK_IMAGE_USAGE_STORAGE_BIT used for raytracing
	create_info.presentMode = present_mode;
	create_info.clipped = VK_TRUE; // If we can't see a pixel, get rid of it
	create_info.oldSwapchain = VK_NULL_HANDLE; // We are not remaking the swapchain
	if (indices.graphics_indices != indices.present_indices)
	{
		create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT; //Image can be used across many queue family's
		create_info.queueFamilyIndexCount = 2;
		create_info.pQueueFamilyIndices = std::vector<uint32_t>{ indices.graphics_indices, indices.present_indices }.data();
	}
	else
	{
		create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; // image is owned by one queue family
		create_info.queueFamilyIndexCount = 0;
		create_info.pQueueFamilyIndices = nullptr;
	}
	create_info.preTransform = swap_chain_support.capabilities.currentTransform; // We don't want any transformations applied to the images
	create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // Ignore the alpha channel so we can do blending

	return create_info;
}

VkSubpassDependency Renderer::Vulkan::VulkanInitializers::SubpassDependency()
{
	VkSubpassDependency subpass_dependency = {};
	subpass_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	subpass_dependency.dstSubpass = 0;
	subpass_dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	subpass_dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	subpass_dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	subpass_dependency.srcAccessMask = 0;
	subpass_dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	return subpass_dependency;
}

VkSubpassDependency Renderer::Vulkan::VulkanInitializers::SubpassDependency(uint32_t srcSubpass, uint32_t dstSubpass, VkPipelineStageFlags dstStageMask, VkAccessFlags dstAccessMask, VkPipelineStageFlags srcStageMask, VkAccessFlags srcAccessMask)
{
	VkSubpassDependency subpass_dependency = {};
	subpass_dependency.srcSubpass = srcSubpass;
	subpass_dependency.dstSubpass = dstSubpass;
	subpass_dependency.dstStageMask = dstStageMask;
	subpass_dependency.dstAccessMask = dstAccessMask;
	subpass_dependency.srcStageMask = srcStageMask;
	subpass_dependency.srcAccessMask = srcAccessMask;
	subpass_dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	return subpass_dependency;
}

VkRenderPassCreateInfo Renderer::Vulkan::VulkanInitializers::RenderPassCreateInfo(std::vector<VkAttachmentDescription>& color_attachment, VkSubpassDescription* subpass, uint32_t subpass_count, VkSubpassDependency* subpass_dependency, uint32_t subpass_dependency_count)
{
	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_info.attachmentCount = static_cast<uint32_t>(color_attachment.size());
	render_pass_info.pAttachments = color_attachment.data();
	render_pass_info.subpassCount = subpass_count;
	render_pass_info.pSubpasses = subpass;
	render_pass_info.dependencyCount = subpass_dependency_count;
	render_pass_info.pDependencies = subpass_dependency;
	return render_pass_info;
}

VkAttachmentDescription Renderer::Vulkan::VulkanInitializers::AttachmentDescription(VkFormat format, VkAttachmentStoreOp store_op, VkImageLayout final_layout)
{
	VkAttachmentDescription color_attachment = {};
	color_attachment.format = format;
	color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR ;
	color_attachment.storeOp = store_op;
	color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment.finalLayout = final_layout;//VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	return color_attachment;
}

VkAttachmentDescription Renderer::Vulkan::VulkanInitializers::AttachmentDescription(VkFormat format, VkAttachmentStoreOp store_op, VkImageLayout final_layout, VkAttachmentLoadOp loadOp)
{
	VkAttachmentDescription color_attachment = {};
	color_attachment.format = format;
	color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	color_attachment.loadOp = loadOp;
	color_attachment.storeOp = store_op;
	color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment.finalLayout = final_layout;//VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	return color_attachment;
}

VkAttachmentReference Renderer::Vulkan::VulkanInitializers::AttachmentReference(VkImageLayout layout, uint32_t attachment)
{
	VkAttachmentReference color_attachment_refrence = {};
	color_attachment_refrence.attachment = attachment;
	color_attachment_refrence.layout = layout;
	return color_attachment_refrence;
}

VkSubpassDescription Renderer::Vulkan::VulkanInitializers::SubpassDescription(VkAttachmentReference* color_attachment_refrence, unsigned int color_count, VkAttachmentReference & depth_attachment_ref)
{
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = color_count;
	subpass.pColorAttachments = color_attachment_refrence;
	subpass.pDepthStencilAttachment = &depth_attachment_ref;
	return subpass;
}

VkSubpassDescription Renderer::Vulkan::VulkanInitializers::SubpassDescription(VkAttachmentReference & color_attachment_refrence)
{
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &color_attachment_refrence;
	return subpass;
}

VkFramebufferCreateInfo Renderer::Vulkan::VulkanInitializers::FramebufferCreateInfo(VkExtent2D & swap_chain_extent, std::vector<VkImageView>& attachments, VkRenderPass & render_pass)
{
	VkFramebufferCreateInfo framebuffer_info = {};
	framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	framebuffer_info.renderPass = render_pass;
	framebuffer_info.attachmentCount = static_cast<uint32_t>(attachments.size());
	framebuffer_info.pAttachments = attachments.data();
	framebuffer_info.width = swap_chain_extent.width;
	framebuffer_info.height = swap_chain_extent.height;
	framebuffer_info.layers = 1;
	return framebuffer_info;
}

VkSemaphoreCreateInfo Renderer::Vulkan::VulkanInitializers::SemaphoreCreateInfo()
{
	VkSemaphoreCreateInfo semaphore_info = {};
	semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	return semaphore_info;
}

VkImageViewCreateInfo Renderer::Vulkan::VulkanInitializers::ImageViewCreate(VkImage image, VkFormat format, VkImageAspectFlags aspect_flags)
{
	VkImageViewCreateInfo create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	create_info.image = image;
	create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
	create_info.format = format;
	create_info.components.r = VK_COMPONENT_SWIZZLE_R;
	create_info.components.g = VK_COMPONENT_SWIZZLE_G;
	create_info.components.b = VK_COMPONENT_SWIZZLE_B;
	create_info.components.a = VK_COMPONENT_SWIZZLE_A;
	create_info.subresourceRange.aspectMask = aspect_flags;// VK_IMAGE_ASPECT_COLOR_BIT;
	create_info.subresourceRange.baseMipLevel = 0;
	create_info.subresourceRange.levelCount = 1;
	create_info.subresourceRange.baseArrayLayer = 0;
	create_info.subresourceRange.layerCount = 1;
	return create_info;
}

VkMemoryAllocateInfo Renderer::Vulkan::VulkanInitializers::MemoryAllocateInfo(VkDeviceSize size, uint32_t memory_type_index)
{
	VkMemoryAllocateInfo alloc_info = {};
	alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	alloc_info.allocationSize = size;
	alloc_info.memoryTypeIndex = memory_type_index;
	return alloc_info;
}

VkImageCreateInfo Renderer::Vulkan::VulkanInitializers::ImageCreateInfo(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, uint32_t mip_levels)
{
	VkImageCreateInfo image_info = {};
	image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_info.imageType = VK_IMAGE_TYPE_2D;
	image_info.extent.width = width;
	image_info.extent.height = height;
	image_info.extent.depth = 1;
	image_info.mipLevels = mip_levels;
	image_info.arrayLayers = 1;
	image_info.format = format;
	image_info.tiling = tiling;
	image_info.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
	image_info.usage = usage;
	image_info.samples = VK_SAMPLE_COUNT_1_BIT;
	image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	return image_info;
}

VkImageMemoryBarrier Renderer::Vulkan::VulkanInitializers::ImageMemoryBarrier()
{
	VkImageMemoryBarrier image_memory_barrier{};
	image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	image_memory_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	image_memory_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	return image_memory_barrier;
}

VkImageMemoryBarrier Renderer::Vulkan::VulkanInitializers::ImageMemoryBarrier(VkImage & image, VkFormat & format, VkImageLayout & old_layout, VkImageLayout & new_layout)
{
	VkImageMemoryBarrier barrier = ImageMemoryBarrier();
	barrier.oldLayout = old_layout;
	barrier.newLayout = new_layout;
	barrier.image = image;
	if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
	{
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

		if (format != VK_FORMAT_UNDEFINED && Renderer::Vulkan::VulkanCommon::HasStencilComponent(format))
		{
			barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
		}
	}
	else
	{
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	}
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;

	if (old_layout == VK_IMAGE_LAYOUT_PREINITIALIZED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	}
	else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	}
	else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
	{
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	}
	else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
	{
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	}
	else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_GENERAL)
	{
		barrier.srcAccessMask = 0;
	}
	return barrier;
}

VkBufferCreateInfo Renderer::Vulkan::VulkanInitializers::BufferCreateInfo(VkDeviceSize size, VkBufferUsageFlags usage)
{
	VkBufferCreateInfo buffer_info = {};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size = size;
	buffer_info.usage = usage;
	buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	return buffer_info;
}

VkDescriptorBufferInfo Renderer::Vulkan::VulkanInitializers::DescriptorBufferInfo(VkBuffer buffer, uint32_t size, VkDeviceSize & offset)
{
	VkDescriptorBufferInfo buffer_info = {};
	buffer_info.buffer = buffer;
	buffer_info.offset = offset;
	buffer_info.range = size;
	return buffer_info;
}

VkDescriptorImageInfo Renderer::Vulkan::VulkanInitializers::DescriptorImageInfo(VkSampler sampler, VkImageView imageView, VkImageLayout imageLayout)
{
	VkDescriptorImageInfo descriptorImageInfo{};
	descriptorImageInfo.sampler = sampler;
	descriptorImageInfo.imageView = imageView;
	descriptorImageInfo.imageLayout = imageLayout;
	return descriptorImageInfo;
}

VkDescriptorPoolSize Renderer::Vulkan::VulkanInitializers::DescriptorPoolSize(VkDescriptorType type, unsigned int count)
{
	VkDescriptorPoolSize pool_size = {};
	pool_size.type = type;
	pool_size.descriptorCount = count;
	return pool_size;
}

VkDescriptorSetLayoutBinding Renderer::Vulkan::VulkanInitializers::DescriptorSetLayoutBinding(VkDescriptorType type, VkShaderStageFlags stage_flags, uint32_t binding, unsigned int count)
{
	VkDescriptorSetLayoutBinding layout_bindings = {};
	layout_bindings.binding = binding;
	layout_bindings.descriptorType = type;
	layout_bindings.descriptorCount = count;
	layout_bindings.stageFlags = stage_flags;
	return layout_bindings;
}

VkDescriptorPoolCreateInfo Renderer::Vulkan::VulkanInitializers::DescriptorPoolCreateInfo(std::vector<VkDescriptorPoolSize>& pool_sizes, uint32_t max_sets)
{
	VkDescriptorPoolCreateInfo create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	create_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
	create_info.pPoolSizes = pool_sizes.data();
	create_info.maxSets = max_sets;
	return create_info;
}

VkDescriptorSetLayoutCreateInfo Renderer::Vulkan::VulkanInitializers::DescriptorSetLayoutCreateInfo(std::vector<VkDescriptorSetLayoutBinding>& layout_bindings)
{
	VkDescriptorSetLayoutCreateInfo layout_info = {};
	layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layout_info.bindingCount = (uint32_t)layout_bindings.size();
	layout_info.pBindings = layout_bindings.data();
	return layout_info;
}

VkPipelineLayoutCreateInfo Renderer::Vulkan::VulkanInitializers::PipelineLayoutCreateInfo(std::vector<VkDescriptorSetLayout> & descriptor_set_layout)
{
	VkPipelineLayoutCreateInfo pipeline_layout_info = {};
	pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipeline_layout_info.setLayoutCount = (uint32_t)descriptor_set_layout.size();
	pipeline_layout_info.pSetLayouts = descriptor_set_layout.data();
	pipeline_layout_info.pushConstantRangeCount = 0;
	pipeline_layout_info.pPushConstantRanges = 0;
	return pipeline_layout_info;
}

VkDescriptorSetAllocateInfo Renderer::Vulkan::VulkanInitializers::DescriptorSetAllocateInfo(std::vector<VkDescriptorSetLayout>& layouts, VkDescriptorPool & pool)
{
	VkDescriptorSetAllocateInfo alloc_info = {};
	alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	alloc_info.descriptorPool = pool;
	alloc_info.descriptorSetCount = static_cast<uint32_t>(layouts.size());
	alloc_info.pSetLayouts = layouts.data();
	return alloc_info;
}

VkShaderModuleCreateInfo Renderer::Vulkan::VulkanInitializers::ShaderModuleCreateInfo(const std::vector<char>& code)
{
	VkShaderModuleCreateInfo create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	create_info.codeSize = code.size();
	create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());
	return create_info;
}

VkPipelineShaderStageCreateInfo Renderer::Vulkan::VulkanInitializers::PipelineShaderStageCreateInfo(VkShaderModule & shader, const char * main, VkShaderStageFlagBits flag)
{
	VkPipelineShaderStageCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	info.stage = flag;
	info.module = shader;
	info.pName = main;
	return info;
}

VkComputePipelineCreateInfo Renderer::Vulkan::VulkanInitializers::ComputePipelineCreateInfo(VkPipelineLayout & layout, VkPipelineShaderStageCreateInfo & shader_stage)
{
	VkComputePipelineCreateInfo create_info{};
	create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	create_info.layout = layout;
	create_info.stage = shader_stage;
	return create_info;
}

VkWriteDescriptorSet Renderer::Vulkan::VulkanInitializers::WriteDescriptorSet(VkDescriptorSet d_set, VkDescriptorImageInfo & image_info, VkDescriptorType type, int binding)
{
	VkWriteDescriptorSet descriptor_write = {};
	descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptor_write.dstSet = d_set; // write to this descriptor set.
	descriptor_write.dstBinding = binding; // write to the first, and only binding.
	descriptor_write.dstArrayElement = 0;
	descriptor_write.descriptorType = type; // Type of buffer
	descriptor_write.descriptorCount = 1; // update a single descriptor.
	descriptor_write.pImageInfo = &image_info;
	return descriptor_write;
}

VkWriteDescriptorSet Renderer::Vulkan::VulkanInitializers::WriteDescriptorSet(VkDescriptorSet d_set, VkDescriptorBufferInfo & buffer_info, VkDescriptorType type, int binding)
{
	VkWriteDescriptorSet descriptor_write = {};
	descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptor_write.dstSet = d_set; // write to this descriptor set.
	descriptor_write.dstBinding = binding; // write to the first, and only binding.
	descriptor_write.dstArrayElement = 0;
	descriptor_write.descriptorType = type; // Type of buffer
	descriptor_write.descriptorCount = 1; // update a single descriptor.
	descriptor_write.pBufferInfo = &buffer_info;
	return descriptor_write;
}

VkWriteDescriptorSet Renderer::Vulkan::VulkanInitializers::WriteDescriptorSet(VkDescriptorSet d_set, std::vector<VkWriteDescriptorSetAccelerationStructureNV>& buffer, VkDescriptorType type, int binding)
{
	VkWriteDescriptorSet descriptorWrite = {};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = d_set;
	descriptorWrite.dstBinding = binding;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = type;
	descriptorWrite.descriptorCount = static_cast<uint32_t>(buffer.size());
	descriptorWrite.pBufferInfo = VK_NULL_HANDLE;
	descriptorWrite.pImageInfo = VK_NULL_HANDLE;
	descriptorWrite.pTexelBufferView = VK_NULL_HANDLE;
	descriptorWrite.pNext = VK_NULL_HANDLE;

	static const int offset = offsetof(VkWriteDescriptorSet, pNext);

	VkWriteDescriptorSetAccelerationStructureNV** data = reinterpret_cast<VkWriteDescriptorSetAccelerationStructureNV**>(reinterpret_cast<uint8_t*>(&descriptorWrite) + offset);

	*data = buffer.data();

	return descriptorWrite;
}

VkWriteDescriptorSet Renderer::Vulkan::VulkanInitializers::WriteDescriptorSet(VkDescriptorSet d_set, VkWriteDescriptorSetAccelerationStructureNV& buffer, VkDescriptorType type, int binding)
{
	VkWriteDescriptorSet descriptorWrite = {};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = d_set;
	descriptorWrite.dstBinding = binding;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = type;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pBufferInfo = VK_NULL_HANDLE;
	descriptorWrite.pImageInfo = VK_NULL_HANDLE;
	descriptorWrite.pTexelBufferView = VK_NULL_HANDLE;
	descriptorWrite.pNext = VK_NULL_HANDLE;

	static const int offset = offsetof(VkWriteDescriptorSet, pNext);

	VkWriteDescriptorSetAccelerationStructureNV** data = reinterpret_cast<VkWriteDescriptorSetAccelerationStructureNV**>(reinterpret_cast<uint8_t*>(&descriptorWrite) + offset);

	*data = &buffer;

	return descriptorWrite;
}

VkWriteDescriptorSet Renderer::Vulkan::VulkanInitializers::WriteDescriptorSet(VkDescriptorSet d_set, std::vector<VkDescriptorImageInfo>& buffer, VkDescriptorType type, int binding)
{
	VkWriteDescriptorSet descriptorWrite = {};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = d_set;
	descriptorWrite.dstBinding = binding;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = type;
	descriptorWrite.descriptorCount = static_cast<uint32_t>(buffer.size());
	descriptorWrite.pBufferInfo = VK_NULL_HANDLE;
	descriptorWrite.pImageInfo = VK_NULL_HANDLE;
	descriptorWrite.pTexelBufferView = VK_NULL_HANDLE;
	descriptorWrite.pNext = VK_NULL_HANDLE;

	static const int offset = offsetof(VkWriteDescriptorSet, pImageInfo);

	VkDescriptorImageInfo** data = reinterpret_cast<VkDescriptorImageInfo**>(reinterpret_cast<uint8_t*>(&descriptorWrite) + offset);
	*data = buffer.data();

	return descriptorWrite;
}

VkWriteDescriptorSet Renderer::Vulkan::VulkanInitializers::WriteDescriptorSet(VkDescriptorSet d_set, std::vector<VkDescriptorBufferInfo>& buffer, VkDescriptorType type, int binding)
{
	VkWriteDescriptorSet descriptor_write = {};
	descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptor_write.dstSet = d_set; // write to this descriptor set.
	descriptor_write.dstBinding = binding; // write to the first, and only binding.
	descriptor_write.dstArrayElement = 0;
	descriptor_write.descriptorType = type; // Type of buffer
	descriptor_write.descriptorCount = 1; // update a single descriptor.
	descriptor_write.descriptorCount = static_cast<uint32_t>(buffer.size());
	descriptor_write.pBufferInfo = VK_NULL_HANDLE;
	descriptor_write.pImageInfo = VK_NULL_HANDLE;
	descriptor_write.pTexelBufferView = VK_NULL_HANDLE;
	descriptor_write.pNext = VK_NULL_HANDLE;

	static const int offset = offsetof(VkWriteDescriptorSet, pBufferInfo);

	VkDescriptorBufferInfo** data = reinterpret_cast<VkDescriptorBufferInfo**>(reinterpret_cast<uint8_t*>(&descriptor_write) + offset);
	*data = buffer.data();

	return descriptor_write;
}

 ;
VkFenceCreateInfo Renderer::Vulkan::VulkanInitializers::CreateFenceInfo()
{
	VkFenceCreateInfo fenceCreateInfo = {};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.flags = 0;
	return fenceCreateInfo;
}

VkVertexInputBindingDescription Renderer::Vulkan::VulkanInitializers::VertexInputBinding(uint32_t binding, uint32_t stride, VkVertexInputRate input_rate)
{
	VkVertexInputBindingDescription binding_description = {};
	binding_description.binding = binding;
	binding_description.stride = stride;
	binding_description.inputRate = input_rate;
	return binding_description;
}

VkVertexInputAttributeDescription Renderer::Vulkan::VulkanInitializers::VertexInputAttributeDescription(uint32_t binding, uint32_t location, VkFormat format, uint32_t offset)
{
	VkVertexInputAttributeDescription vertex_description = {};
	vertex_description.binding = binding;
	vertex_description.location = location;
	vertex_description.format = format;
	vertex_description.offset = offset;
	return vertex_description;
}

VkPipelineVertexInputStateCreateInfo Renderer::Vulkan::VulkanInitializers::PipelineVertexInputStateCreateInfo(std::vector<VkVertexInputBindingDescription>& bd, std::vector<VkVertexInputAttributeDescription>& add)
{
	VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
	vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertex_input_info.vertexBindingDescriptionCount = static_cast<uint32_t>(bd.size());
	vertex_input_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(add.size());
	vertex_input_info.pVertexBindingDescriptions = bd.data();
	vertex_input_info.pVertexAttributeDescriptions = add.data();
	return vertex_input_info;
}

VkPipelineViewportStateCreateInfo Renderer::Vulkan::VulkanInitializers::PipelineViewportStateCreateInfo(uint32_t viewport_count, uint32_t scissor_count)
{
	VkPipelineViewportStateCreateInfo viewport_state = {};
	viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport_state.viewportCount = viewport_count;
	viewport_state.scissorCount = scissor_count;
	return viewport_state;
}

VkPipelineRasterizationStateCreateInfo Renderer::Vulkan::VulkanInitializers::PipelineRasterizationStateCreateInfo(VkCullModeFlagBits cull_mode, VkFrontFace front_face, VkPolygonMode poly_mode, float line_width)
{
	VkPipelineRasterizationStateCreateInfo rasterizer = {};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = poly_mode;// VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = line_width;
	rasterizer.cullMode = cull_mode;
	rasterizer.frontFace = front_face;
	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthBiasConstantFactor = 0.0f;
	rasterizer.depthBiasClamp = 0.0f;
	rasterizer.depthBiasSlopeFactor = 0.0f;
	return rasterizer;
}

VkPipelineMultisampleStateCreateInfo Renderer::Vulkan::VulkanInitializers::PipelineMultisampleStateCreateInfo()
{
	VkPipelineMultisampleStateCreateInfo multisampling = {};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampling.minSampleShading = 1.0f;
	multisampling.pSampleMask = nullptr;
	multisampling.alphaToCoverageEnable = VK_FALSE;
	multisampling.alphaToOneEnable = VK_FALSE;
	return multisampling;
}

VkPipelineDepthStencilStateCreateInfo Renderer::Vulkan::VulkanInitializers::PipelineDepthStencilStateCreateInfo(bool enable_depth)
{
	VkPipelineDepthStencilStateCreateInfo depth_stencil = {};
	depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depth_stencil.depthTestEnable = enable_depth;
	depth_stencil.depthWriteEnable = enable_depth;
	depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
	depth_stencil.depthBoundsTestEnable = VK_FALSE;
	depth_stencil.stencilTestEnable = VK_FALSE;
	return depth_stencil;
}

VkPipelineColorBlendAttachmentState Renderer::Vulkan::VulkanInitializers::PipelineColorBlendAttachmentState()
{
	VkPipelineColorBlendAttachmentState color_blend_attachment = {};
	color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	color_blend_attachment.blendEnable = VK_TRUE;
	color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
	color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
	return color_blend_attachment;
}

VkPipelineDynamicStateCreateInfo Renderer::Vulkan::VulkanInitializers::PipelineDynamicStateCreateInfo(const std::vector<VkDynamicState>& states)
{
	VkPipelineDynamicStateCreateInfo state = {};
	state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	state.pDynamicStates = states.data();
	state.dynamicStateCount = static_cast<uint32_t>(states.size());
	state.flags = 0;
	return state;
}

VkPipelineColorBlendStateCreateInfo Renderer::Vulkan::VulkanInitializers::PipelineColorBlendStateCreateInfo(VkPipelineColorBlendAttachmentState & color_blend_attachment)
{
	VkPipelineColorBlendStateCreateInfo color_blending = {};
	color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	color_blending.logicOpEnable = VK_FALSE;
	color_blending.logicOp = VK_LOGIC_OP_COPY;
	color_blending.attachmentCount = 1;
	color_blending.pAttachments = &color_blend_attachment;
	color_blending.blendConstants[0] = 0.0f;
	color_blending.blendConstants[1] = 0.0f;
	color_blending.blendConstants[2] = 0.0f;
	color_blending.blendConstants[3] = 0.0f;
	return color_blending;
}

VkPipelineInputAssemblyStateCreateInfo Renderer::Vulkan::VulkanInitializers::PipelineInputAssemblyStateCreateInfo(VkPrimitiveTopology tpology)
{
	VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
	input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	input_assembly.topology = tpology;
	input_assembly.primitiveRestartEnable = VK_FALSE;
	return input_assembly;
}

VkGraphicsPipelineCreateInfo Renderer::Vulkan::VulkanInitializers::GraphicsPipelineCreateInfo(std::vector<VkPipelineShaderStageCreateInfo>& shader_stages, VkPipelineVertexInputStateCreateInfo & vertex_input_info, VkPipelineInputAssemblyStateCreateInfo & input_assembly, VkPipelineViewportStateCreateInfo & viewport_state, VkPipelineRasterizationStateCreateInfo & rasterizer, VkPipelineMultisampleStateCreateInfo & multisampling, VkPipelineColorBlendStateCreateInfo & color_blending, VkPipelineDepthStencilStateCreateInfo & depth_stencil, VkPipelineLayout & layout, VkRenderPass & render_pass, VkPipelineDynamicStateCreateInfo & dynamic_state)
{
	VkGraphicsPipelineCreateInfo pipeline_info = {};
	pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipeline_info.stageCount = static_cast<uint32_t>(shader_stages.size());
	pipeline_info.pStages = shader_stages.data();

	pipeline_info.pVertexInputState = &vertex_input_info;
	pipeline_info.pInputAssemblyState = &input_assembly;
	pipeline_info.pViewportState = &viewport_state;
	pipeline_info.pRasterizationState = &rasterizer;
	pipeline_info.pMultisampleState = &multisampling;
	pipeline_info.pDepthStencilState = &depth_stencil;
	pipeline_info.pColorBlendState = &color_blending;
	pipeline_info.pDynamicState = &dynamic_state;
	pipeline_info.layout = layout;
	pipeline_info.renderPass = render_pass;
	pipeline_info.subpass = 0;
	pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
	pipeline_info.basePipelineIndex = -1;
	pipeline_info.flags = VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT;
	return pipeline_info;
}

VkRect2D Renderer::Vulkan::VulkanInitializers::Rect2D(int32_t width, int32_t height, int32_t x_offset, int32_t y_offset)
{
	VkRect2D rec{};
	rec.extent.width = width;
	rec.extent.height = height;
	rec.offset.x = x_offset;
	rec.offset.y = y_offset;
	return rec;
}

VkViewport Renderer::Vulkan::VulkanInitializers::Viewport(float width, float height, float x, float y, float min, float max)
{
	VkViewport viewport = {};
	viewport.x = x;
	viewport.y = y;
	viewport.width = width;
	viewport.height = height;
	viewport.minDepth = min;
	viewport.maxDepth = max;
	return viewport;
}

VkRect2D Renderer::Vulkan::VulkanInitializers::Scissor(uint32_t width, uint32_t height)
{
	VkRect2D dim = Rect2D(width, height, 0, 0);
	dim.extent.width = width;
	dim.extent.height = height;
	return dim;
}

VkSamplerCreateInfo Renderer::Vulkan::VulkanInitializers::SamplerCreateInfo()
{
	VkSamplerCreateInfo sampler_create_info{};
	sampler_create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler_create_info.maxAnisotropy = 1.0f;
	return sampler_create_info;
}

VkRayTracingShaderGroupCreateInfoNV Renderer::Vulkan::VulkanInitializers::RayTracingShaderGroupCreateNV(VkRayTracingShaderGroupTypeNV type)
{
	VkRayTracingShaderGroupCreateInfoNV groupInfo{};
	groupInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
	groupInfo.pNext = nullptr;
	groupInfo.type = type;
	groupInfo.generalShader = VK_SHADER_UNUSED_NV;
	groupInfo.closestHitShader = VK_SHADER_UNUSED_NV;
	groupInfo.anyHitShader = VK_SHADER_UNUSED_NV;
	groupInfo.intersectionShader = VK_SHADER_UNUSED_NV;
	return groupInfo;
}

VkRayTracingPipelineCreateInfoNV Renderer::Vulkan::VulkanInitializers::RayTracePipelineCreateInfoNV(std::vector<VkPipelineShaderStageCreateInfo>& shader_stages, std::vector<VkRayTracingShaderGroupCreateInfoNV>& groups, VkPipelineLayout layout, uint32_t maxRecursion)
{
	VkRayTracingPipelineCreateInfoNV pipeline{};
	pipeline.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV;
	pipeline.pNext = nullptr;
	pipeline.flags = 0;
	pipeline.stageCount = static_cast<uint32_t>(shader_stages.size());
	pipeline.pStages = shader_stages.data();
	pipeline.groupCount = static_cast<uint32_t>(groups.size());
	pipeline.pGroups = groups.data();
	pipeline.maxRecursionDepth = maxRecursion;
	pipeline.layout = layout;
	pipeline.basePipelineHandle = VK_NULL_HANDLE;
	pipeline.basePipelineIndex = 0;
	return pipeline;
}

VkGeometryNV Renderer::Vulkan::VulkanInitializers::CreateRayTraceGeometry(VkBuffer vertexBuffer, VkDeviceSize vertexOffsetInBytes, uint32_t vertexCount, VkDeviceSize vertexSizeInBytes, VkBuffer indexBuffer, VkDeviceSize indexOffsetInBytes, uint32_t indexCount, VkBuffer transformBuffer, VkDeviceSize transformOffsetInBytes, bool isOpaque)
{
	VkGeometryNV geometry;
	geometry.sType = VK_STRUCTURE_TYPE_GEOMETRY_NV;
	geometry.pNext = nullptr;
	geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_NV;
	geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV;
	geometry.geometry.triangles.pNext = nullptr;
	geometry.geometry.triangles.vertexData = vertexBuffer;
	geometry.geometry.triangles.vertexOffset = vertexOffsetInBytes;
	geometry.geometry.triangles.vertexCount = vertexCount;
	geometry.geometry.triangles.vertexStride = vertexSizeInBytes;
	// Limitation to 3xfloat32 for vertices
	geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	geometry.geometry.triangles.indexData = indexBuffer;
	geometry.geometry.triangles.indexOffset = indexOffsetInBytes;
	geometry.geometry.triangles.indexCount = indexCount;
	// Limitation to 32-bit indices
	geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
	geometry.geometry.triangles.transformData = transformBuffer;
	geometry.geometry.triangles.transformOffset = transformOffsetInBytes;
	geometry.geometry.aabbs = { VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV };
	geometry.flags = isOpaque ? VK_GEOMETRY_OPAQUE_BIT_NV : 0;

	return geometry;
}

VkAccelerationStructureInfoNV Renderer::Vulkan::VulkanInitializers::AccelerationStructureInfoNV(VkAccelerationStructureTypeNV type, VkBuildAccelerationStructureFlagsNV flags, const VkGeometryNV* prt, uint32_t count,uint32_t instance_count)
{
	VkAccelerationStructureInfoNV accelerationStructureInfo{};
	accelerationStructureInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
	accelerationStructureInfo.type = type;
	accelerationStructureInfo.flags = flags;
	accelerationStructureInfo.instanceCount = instance_count;  // The bottom-level AS can only contain explicit geometry, and no instances
	accelerationStructureInfo.geometryCount = count;
	accelerationStructureInfo.pGeometries = prt;

	return accelerationStructureInfo;
}

VkAccelerationStructureCreateInfoNV Renderer::Vulkan::VulkanInitializers::AccelerationStructureCreateInfoNV(VkAccelerationStructureInfoNV structure_info)
{
	VkAccelerationStructureCreateInfoNV info{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV };
	info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
	info.pNext = nullptr;
	info.info = structure_info;
	info.compactedSize = 0;
	return info;
}

VkAccelerationStructureMemoryRequirementsInfoNV Renderer::Vulkan::VulkanInitializers::AccelerationStructureMemoryRequirmentsInfoNV(VkAccelerationStructureNV str)
{
	VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo;
	memoryRequirementsInfo.sType =
		VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
	memoryRequirementsInfo.pNext = nullptr;
	memoryRequirementsInfo.accelerationStructure = str;
	memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
	return memoryRequirementsInfo;
}

VkBindAccelerationStructureMemoryInfoNV Renderer::Vulkan::VulkanInitializers::AccelerationStructureMemoryInfoNV(VkAccelerationStructureNV str, VkDeviceMemory memory)
{
	VkBindAccelerationStructureMemoryInfoNV info{};
	info.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
	info.pNext = nullptr;
	info.accelerationStructure = str;
	info.memory = memory;
	info.memoryOffset = 0;
	info.deviceIndexCount = 0;
	info.pDeviceIndices = nullptr;
	return info;
}

VkAccelerationStructureInfoNV Renderer::Vulkan::VulkanInitializers::AccelerationStructureInfo(VkBuildAccelerationStructureFlagsNV flags, std::vector<VkGeometryNV>& buffer)
{
	VkAccelerationStructureInfoNV info;
	info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
	info.pNext = nullptr;
	info.flags = flags;
	info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
	info.geometryCount = static_cast<uint32_t>(buffer.size());
	info.pGeometries = buffer.data();
	info.instanceCount = 0;
	return info;
}

VkAccelerationStructureInfoNV Renderer::Vulkan::VulkanInitializers::AccelerationStructureInfo(VkBuildAccelerationStructureFlagsNV flags, unsigned int instanceCount)
{
	VkAccelerationStructureInfoNV info;
	info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
	info.pNext = nullptr;
	info.flags = flags;
	info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
	info.geometryCount = 0;
	info.pGeometries = nullptr;
	info.instanceCount = instanceCount;
	return info;
}

VkWriteDescriptorSetAccelerationStructureNV Renderer::Vulkan::VulkanInitializers::WriteDescriptorSetAccelerator(VkAccelerationStructureNV& acceleration)
{
	VkWriteDescriptorSetAccelerationStructureNV descriptorAccelerationStructureInfo;
	descriptorAccelerationStructureInfo.sType =
		VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV;
	descriptorAccelerationStructureInfo.pNext = nullptr;
	descriptorAccelerationStructureInfo.accelerationStructureCount = 1;
	descriptorAccelerationStructureInfo.pAccelerationStructures = &acceleration;
	return descriptorAccelerationStructureInfo;
}


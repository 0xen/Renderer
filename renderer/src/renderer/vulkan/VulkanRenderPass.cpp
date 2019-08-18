#include <renderer/vulkan/VulkanRenderPass.hpp>
#include <renderer/vulkan/VulkanSwapchain.hpp>
#include <renderer/vulkan/VulkanPhysicalDevice.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanInstance.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>
#include <renderer/vulkan/VulkanGraphicsPipeline.hpp>
#include <renderer/vulkan/VulkanRaytracePipeline.hpp>
#include <renderer/vulkan/VulkanRenderer.hpp>
#include <renderer/vulkan/VulkanDescriptorPool.hpp>
#include <renderer/vulkan/VulkanDescriptorSet.hpp>

#include <assert.h>

Renderer::Vulkan::VulkanRenderPass::VulkanRenderPass(VulkanRenderer* renderer, VulkanSwapchain* swapchain, VulkanInstance* instance, VulkanDevice* device, unsigned int subpass_count)
{
	m_renderer = renderer;
	m_instance = instance;
	m_device = device;
	m_swapchain = swapchain;
	m_subpass_count = subpass_count;

	CreateRenderPass();
	InitFences();

	m_should_rebuild_cmd = true;


}

Renderer::Vulkan::VulkanRenderPass::~VulkanRenderPass()
{
	DestroyRenderPass();
}

void Renderer::Vulkan::VulkanRenderPass::AttachGraphicsPipeline(VulkanGraphicsPipeline * pipeline)
{
	// Attach the pipeline to its correct sub pass
	m_subpasses[pipeline->GetGraphicsPipelineConfig().subpass].push_back(pipeline);
}

void Renderer::Vulkan::VulkanRenderPass::RemoveGraphicsPipeline(VulkanGraphicsPipeline * pipeline)
{
	auto it = std::find(m_subpasses[pipeline->GetGraphicsPipelineConfig().subpass].begin(), m_subpasses[pipeline->GetGraphicsPipelineConfig().subpass].end(), pipeline);
	if (it != m_subpasses[pipeline->GetGraphicsPipelineConfig().subpass].end())
	{
		m_subpasses[pipeline->GetGraphicsPipelineConfig().subpass].erase(it);
	}
}

void Renderer::Vulkan::VulkanRenderPass::Rebuild()
{
	DestroyRenderPass();
	CreateRenderPass();
	RebuildCommandBuffers();
}

void Renderer::Vulkan::VulkanRenderPass::RebuildCommandBuffers()
{
	VkCommandBufferBeginInfo begin_info = VulkanInitializers::CommandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
	std::vector<VkClearValue> clear_values;
	clear_values.resize(4);
	Renderer::NativeWindowHandle* window_handle = m_swapchain->GetNativeWindowHandle();
	std::copy(std::begin(window_handle->clear_color.float32), std::end(window_handle->clear_color.float32), std::begin(clear_values[0].color.float32));
	std::copy(std::begin(window_handle->clear_color.float32), std::end(window_handle->clear_color.float32), std::begin(clear_values[1].color.float32));
	std::copy(std::begin(window_handle->clear_color.float32), std::end(window_handle->clear_color.float32), std::begin(clear_values[2].color.float32));
	clear_values[3].depthStencil = { 1.0f, 0 };
	VkRenderPassBeginInfo render_pass_info = VulkanInitializers::RenderPassBeginInfo(m_render_pass, m_swapchain->GetSwapchainExtent(), clear_values);

	VkImageSubresourceRange subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

	for (uint32_t i = 0; i < m_command_buffers.size(); i++)
	{

		// Reset the command buffers
		vkResetCommandBuffer(
			m_command_buffers[i],
			0
		);

		// Setup unique frame buffer
		render_pass_info.framebuffer = m_swap_chain_framebuffers[i];

		ErrorCheck(vkBeginCommandBuffer(
			m_command_buffers[i],
			&begin_info
		));

		assert(!HasError() && "Unable to create command buffer");

		if ((m_instance->GetFlags()&VulkanFlags::Raytrace) == VulkanFlags::Raytrace)
		{
			for (auto pipeline : m_subpasses[0])
			{
				if (dynamic_cast<VulkanRaytracePipeline*>(pipeline) != nullptr)
				{
					pipeline->AttachToCommandBuffer(m_command_buffers[i]);
				}
			}
			VkImage target_image = m_swapchain->GetSwapchainImages()[i];
			VkImageLayout target_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
			if(m_subpass_count > 1)
			{
				target_image = m_attachments[i].color1.image;
				target_layout = VK_IMAGE_LAYOUT_GENERAL;
			}

			VulkanCommon::TransitionImageLayout(m_command_buffers[i], target_image, VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, subresourceRange);


			VulkanCommon::TransitionImageLayout(m_command_buffers[i], m_swapchain->GetRaytracingScratchImage(), VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, subresourceRange);


			VkImageCopy copyRegion{};
			copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			copyRegion.srcOffset = { 0, 0, 0 };
			copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			copyRegion.dstOffset = { 0, 0, 0 };
			copyRegion.extent = { (unsigned int)window_handle->width, (unsigned int)window_handle->height, 1 };
			vkCmdCopyImage(m_command_buffers[i], m_swapchain->GetRaytracingScratchImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);


			VulkanCommon::TransitionImageLayout(m_command_buffers[i], target_image, VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, target_layout,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, subresourceRange);


			VulkanCommon::TransitionImageLayout(m_command_buffers[i], m_swapchain->GetRaytracingScratchImage(), VK_FORMAT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, subresourceRange);
		}

		vkCmdBeginRenderPass(
			m_command_buffers[i],
			&render_pass_info,
			VK_SUBPASS_CONTENTS_INLINE
		);


		vkCmdSetLineWidth(m_command_buffers[i], 1.0f);
		const VkViewport viewport = VulkanInitializers::Viewport((float)window_handle->width, (float)window_handle->height, 0.0f, 0.0f, 0.0f, 1.0f);
		const VkRect2D scissor = VulkanInitializers::Scissor(window_handle->width, window_handle->height);
		vkCmdSetViewport(m_command_buffers[i], 0, 1, &viewport);
		vkCmdSetScissor(m_command_buffers[i], 0, 1, &scissor);


		for (auto pipeline : m_subpasses[0])
		{
			if (dynamic_cast<VulkanRaytracePipeline*>(pipeline) == nullptr)
			{
				pipeline->AttachToCommandBuffer(m_command_buffers[i]);
			}
		}



		for (int subpass = 1; subpass < m_subpass_count && m_subpasses.find(subpass) != m_subpasses.end(); subpass++)
		{
			vkCmdNextSubpass(m_command_buffers[i], VK_SUBPASS_CONTENTS_INLINE);


			for (VulkanGraphicsPipeline* pipeline : m_subpasses[subpass])
			{

				pipeline->AttachPipeline(m_command_buffers[i]);

				if (pipeline->GetGraphicsPipelineConfig().input == ATTACHMENT)
				{
					vkCmdBindDescriptorSets(
						m_command_buffers[i],
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						pipeline->GetPipelineLayout(),
						0,
						1,
						&m_input_attachments_read_sets[i][(subpass + 1) % 2]->GetDescriptorSet(),
						0,
						NULL
					);
				}
				else if (pipeline->GetGraphicsPipelineConfig().input == COMBINED_IMAGE_SAMPLER)
				{
					vkCmdBindDescriptorSets(
						m_command_buffers[i],
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						pipeline->GetPipelineLayout(),
						0,
						1,
						&m_combined_image_sampler_read_sets[i][(subpass + 1) % 2]->GetDescriptorSet(),
						0,
						NULL
					);
				}
				else
				{
					assert(0 && "Invalid pipeline input");
				}


				pipeline->BindDescriptorSets(m_command_buffers[i]);
				pipeline->RenderModels(m_command_buffers[i]);

			}
		}

		vkCmdEndRenderPass(
			m_command_buffers[i]
		);







		ErrorCheck(vkEndCommandBuffer(
			m_command_buffers[i]
		));

		assert(!HasError() && "Unable to end command buffer");

	}
}

void Renderer::Vulkan::VulkanRenderPass::RequestRebuildCommandBuffers()
{
	m_should_rebuild_cmd = true;
}

void Renderer::Vulkan::VulkanRenderPass::Render()
{
	FindNextImageIndex();

	// Get the frame we are rendering too
	unsigned int current_frame_index = GetCurrentFrameIndex();

	SubmitQueue(current_frame_index);

	std::vector<VkSemaphore> signal_semaphores{
		m_swapchain->GetRenderFinishedSemaphore()
	};

	m_swapchain->Present(signal_semaphores, current_frame_index);
}

void Renderer::Vulkan::VulkanRenderPass::FindNextImageIndex()
{
	for (auto subpass : m_subpasses)
	{
		for (auto pipeline : subpass.second)
		{
			if (pipeline->HasChanged())
			{
				// Even when found, keep looping, as if we have multiple pipelines that have changed, we want to reset all there changes flags
				m_should_rebuild_cmd = true;
			}
		}
	}

	if (m_should_rebuild_cmd)
	{
		m_should_rebuild_cmd = false;
		RebuildCommandBuffers();
	}
	vkWaitForFences(*m_device->GetVulkanDevice(), 1, &m_fence[m_frame_index], VK_TRUE, 100);

	VkResult check = vkAcquireNextImageKHR(
		*m_device->GetVulkanDevice(),
		m_swapchain->GetSwapchain(),
		UINT64_MAX,
		m_swapchain->GetImageAvailableSemaphore(),
		VK_NULL_HANDLE,
		&m_frame_index
	);
	if (check == VK_ERROR_OUT_OF_DATE_KHR || m_should_rebuild_cmd)
	{
		m_swapchain->RebuildSwapchain();
		//GetCurrentFrameIndex();
		return;
	}
	ErrorCheck(check);
	assert(!HasError());
	vkQueueWaitIdle(
		*m_device->GetPresentQueue()
	);
}

unsigned int Renderer::Vulkan::VulkanRenderPass::GetCurrentFrameIndex()
{
	return m_frame_index;
}

Renderer::Vulkan::VulkanDescriptorPool* Renderer::Vulkan::VulkanRenderPass::GetInputAttachmentsReadPool()
{
	return m_input_attachments_read_pool;
}

Renderer::Vulkan::VulkanDescriptorPool * Renderer::Vulkan::VulkanRenderPass::GetCombinedImageSamplerReadPool()
{
	return m_combined_image_sampler_read_pool;
}

void Renderer::Vulkan::VulkanRenderPass::SubmitQueue(unsigned int currentBuffer)
{
	// Since we are only dealing with one buffer, select it
	if ((m_instance->GetFlags() & VulkanFlags::ActiveCMDRebuild) == VulkanFlags::ActiveCMDRebuild) currentBuffer = 0;

	VkSubmitInfo info = m_swapchain->GetSubmitInfo();
	info.pCommandBuffers = &m_command_buffers[currentBuffer];

	vkResetFences(*m_device->GetVulkanDevice(), 1, &m_fence[currentBuffer]);

	ErrorCheck(vkQueueSubmit(
		*m_device->GetGraphicsQueue(),
		1,
		&info,
		m_fence[currentBuffer]
	));

	assert(!HasError());
}

VkRenderPass& Renderer::Vulkan::VulkanRenderPass::GetRenderPass()
{
	return m_render_pass;
}

void Renderer::Vulkan::VulkanRenderPass::CreateRenderPass()
{
	InitRenderPass();
	InitFrameBuffer();
	InitCommandBuffers();
}

void Renderer::Vulkan::VulkanRenderPass::DestroyRenderPass()
{
	DeInitFrameBuffer();
	DeInitRenderPass();
}

void Renderer::Vulkan::VulkanRenderPass::InitRenderPass()
{

	const VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;


	m_attachments.resize(m_swapchain->GetSwapchainImages().size());
	m_input_attachments_read_sets.resize(m_attachments.size());
	m_combined_image_sampler_read_sets.resize(m_attachments.size());

	// Read instance for input attachments
	m_input_attachments_read_pool = m_renderer->CreateDescriptorPool({
		m_renderer->CreateDescriptor(VkDescriptorType::VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT, 0), // Color
		m_renderer->CreateDescriptor(VkDescriptorType::VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT, 1), // Depth
		});


	// Read instance for combined image samplers
	m_combined_image_sampler_read_pool = m_renderer->CreateDescriptorPool({
		m_renderer->CreateDescriptor(VkDescriptorType::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT, 0), // Color
		m_renderer->CreateDescriptor(VkDescriptorType::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT, 1), // Depth
		});

	for (int i = 0; i < m_attachments.size(); i++)
	{
		// Create the images and make it to that they are readable as color attachments and as image samplers
		CreateAttachmentImages(colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, m_attachments[i].color1);
		CreateAttachmentImages(colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, m_attachments[i].color2);
		CreateAttachmentImages(VulkanCommon::GetDepthImageFormat(m_device), VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, m_attachments[i].depth);

		{
			// Input attachments
			{
				VulkanDescriptorSet* set = m_input_attachments_read_pool->CreateDescriptorSet();
				set->AttachBuffer(0, VulkanInitializers::DescriptorImageInfo(VK_NULL_HANDLE, m_attachments[i].color1.view, VK_IMAGE_LAYOUT_GENERAL));
				set->AttachBuffer(1, VulkanInitializers::DescriptorImageInfo(VK_NULL_HANDLE, m_attachments[i].depth.view, VK_IMAGE_LAYOUT_GENERAL));
				set->UpdateSet();
				m_input_attachments_read_sets[i][0] = set;
			}


			{
				VulkanDescriptorSet* set = m_input_attachments_read_pool->CreateDescriptorSet();
				set->AttachBuffer(0, VulkanInitializers::DescriptorImageInfo(VK_NULL_HANDLE, m_attachments[i].color2.view, VK_IMAGE_LAYOUT_GENERAL));
				set->AttachBuffer(1, VulkanInitializers::DescriptorImageInfo(VK_NULL_HANDLE, m_attachments[i].depth.view, VK_IMAGE_LAYOUT_GENERAL));
				set->UpdateSet();
				m_input_attachments_read_sets[i][1] = set;
			}
		}
		{
			// Combined image samplers
			{
				VulkanDescriptorSet* set = m_combined_image_sampler_read_pool->CreateDescriptorSet();
				set->AttachBuffer(0, VulkanInitializers::DescriptorImageInfo(m_attachments[i].color1.sampler, m_attachments[i].color1.view, VK_IMAGE_LAYOUT_GENERAL));
				set->AttachBuffer(1, VulkanInitializers::DescriptorImageInfo(m_attachments[i].depth.sampler, m_attachments[i].depth.view, VK_IMAGE_LAYOUT_GENERAL));
				set->UpdateSet();
				m_combined_image_sampler_read_sets[i][0] = set;
			}


			{
				VulkanDescriptorSet* set = m_combined_image_sampler_read_pool->CreateDescriptorSet();
				set->AttachBuffer(0, VulkanInitializers::DescriptorImageInfo(m_attachments[i].color2.sampler, m_attachments[i].color2.view, VK_IMAGE_LAYOUT_GENERAL));
				set->AttachBuffer(1, VulkanInitializers::DescriptorImageInfo(m_attachments[i].depth.sampler, m_attachments[i].depth.view, VK_IMAGE_LAYOUT_GENERAL));
				set->UpdateSet();
				m_combined_image_sampler_read_sets[i][1] = set;
			}
		}
	}



	std::vector<VkAttachmentDescription> attachments;
	if ((m_instance->GetFlags() & VulkanFlags::Raytrace) == VulkanFlags::Raytrace)
	{
		// Needs sorting for raytracing
		attachments = {
			// Define VK_ATTACHMENT_LOAD_OP_DONT_CARE, this will stop the images from being cleared
			VulkanInitializers::AttachmentDescription(m_swapchain->GetSwapChainImageFormat(), VK_ATTACHMENT_STORE_OP_STORE,VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ATTACHMENT_LOAD_OP_DONT_CARE) ,	//Present
			VulkanInitializers::AttachmentDescription(colorFormat, VK_ATTACHMENT_STORE_OP_DONT_CARE,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_DONT_CARE),	//Color
			VulkanInitializers::AttachmentDescription(colorFormat, VK_ATTACHMENT_STORE_OP_DONT_CARE,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_DONT_CARE),	//Color
			VulkanInitializers::AttachmentDescription(VulkanCommon::GetDepthImageFormat(m_device), VK_ATTACHMENT_STORE_OP_DONT_CARE,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)		// Depth
		};
	}
	else
	{
		attachments = {
			VulkanInitializers::AttachmentDescription(m_swapchain->GetSwapChainImageFormat(), VK_ATTACHMENT_STORE_OP_STORE,VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),	// Present
			VulkanInitializers::AttachmentDescription(colorFormat, VK_ATTACHMENT_STORE_OP_DONT_CARE,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),	//Color
			VulkanInitializers::AttachmentDescription(colorFormat, VK_ATTACHMENT_STORE_OP_DONT_CARE,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),	//Color
			VulkanInitializers::AttachmentDescription(VulkanCommon::GetDepthImageFormat(m_device), VK_ATTACHMENT_STORE_OP_DONT_CARE,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)		// Depth
		};
	}

	std::vector<VkSubpassDependency> subpass_dependency;

	std::vector<VkSubpassDescription> subpasses_descriptions;
	// Attachment reference to the final swapchain image
	VkAttachmentReference color_attachment_refrence_swapchain = VulkanInitializers::AttachmentReference(VK_IMAGE_LAYOUT_GENERAL, 0);
	// Attachment reference to the scratch images
	VkAttachmentReference color_attachment_refrence[2] = { VulkanInitializers::AttachmentReference(VK_IMAGE_LAYOUT_GENERAL, 1),VulkanInitializers::AttachmentReference(VK_IMAGE_LAYOUT_GENERAL, 2) };
	// Attachment reference to the depth image
	VkAttachmentReference depth_attachment_refrence = VulkanInitializers::AttachmentReference(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 3);



	if (m_subpass_count == 1)
	{
		// SubPass 0
		// Wright directly to the swapchain
		VkSubpassDescription subpass = VulkanInitializers::SubpassDescription(&color_attachment_refrence_swapchain, 1, depth_attachment_refrence);
		subpasses_descriptions.push_back(subpass);

		// Define subpass dependency, Should just pass data to the swapchain
		subpass_dependency.push_back(VulkanInitializers::SubpassDependency());
	}
	else
	{
		// SubPass 0
		{
			VkSubpassDescription subpass = VulkanInitializers::SubpassDescription(&color_attachment_refrence[0], 1, depth_attachment_refrence);
			subpasses_descriptions.push_back(subpass);

			// Define subpass dependency, Should pas data to scratch iamge
			subpass_dependency.push_back(
				VulkanInitializers::SubpassDependency(
					VK_SUBPASS_EXTERNAL,
					0,
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
					VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
					VK_ACCESS_MEMORY_READ_BIT
				));
		}

		// Define that these sub passes will be taking a direct input from the previous ones
		VkAttachmentReference inputReferences[2][2];
		inputReferences[0][0] = { 1, VK_IMAGE_LAYOUT_GENERAL }; // Storage image 1
		inputReferences[0][1] = { 3, VK_IMAGE_LAYOUT_GENERAL }; // Depth buffer
		inputReferences[1][0] = { 2, VK_IMAGE_LAYOUT_GENERAL }; // Storage image 2
		inputReferences[1][1] = { 3, VK_IMAGE_LAYOUT_GENERAL }; // Depth buffer



		// Loop through for all sub passes excluding the first and last one
		for (int i = 1; i < m_subpass_count - 1; i++)
		{
			// Define that we will be writing to the scratch image
			VkSubpassDescription subpass = VulkanInitializers::SubpassDescription(color_attachment_refrence[i % 2]);

			// Use the attachments filled in the first pass as input attachments
			subpass.inputAttachmentCount = 2;
			subpass.pInputAttachments = inputReferences[(i + 1) % 2];

			subpasses_descriptions.push_back(subpass);


			// Both happen at the same time
			// First transitions the input from color to shader read
			subpass_dependency.push_back(
				VulkanInitializers::SubpassDependency(
					i - 1,
					i,
					VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
					VK_ACCESS_SHADER_READ_BIT,
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
				));

			// Second reads the data and works with it
			subpass_dependency.push_back(
				VulkanInitializers::SubpassDependency(
					i - 1,
					i + 1,
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
				));


		}

		// Attach the final subpass
		{
			// Point to the swapchain as the color refrence
			VkSubpassDescription subpass = VulkanInitializers::SubpassDescription(color_attachment_refrence_swapchain);


			// Use the attachments filled in the first pass as input attachments
			subpass.inputAttachmentCount = 2;
			subpass.pInputAttachments = inputReferences[(m_subpass_count) % 2];

			subpasses_descriptions.push_back(subpass);


			// Both happen at the same time
			// First transitions the input from color to shader read
			subpass_dependency.push_back(
				VulkanInitializers::SubpassDependency(
					m_subpass_count - 2,
					m_subpass_count - 1,
					VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
					VK_ACCESS_SHADER_READ_BIT,
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
				));

			// Second reads the data and works with it
			subpass_dependency.push_back(
				VulkanInitializers::SubpassDependency(
					m_subpass_count - 2,
					VK_SUBPASS_EXTERNAL,
					VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
					VK_ACCESS_MEMORY_READ_BIT,
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
				));
		}

	}


	


	//std::vector<VkSubpassDependency> subpass_dependency = {
	//	VulkanInitializers::SubpassDependency(
	//		VK_SUBPASS_EXTERNAL,
	//		0,
	//		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
	//		VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	//		VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
	//		VK_ACCESS_MEMORY_READ_BIT
	//	),

	//	// Both happen at the same time
	//	// First transitions the input from color to shader read
	//	VulkanInitializers::SubpassDependency(
	//		0,
	//		1,
	//		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	//		VK_ACCESS_SHADER_READ_BIT,
	//		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
	//		VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
	//	),
	//	// Second reads the data and works with it
	//	VulkanInitializers::SubpassDependency(
	//		0,
	//		2,
	//		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
	//		VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	//		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
	//		VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
	//	),


	//	// Both happen at the same time
	//	// First transitions the input from color to shader read
	//	VulkanInitializers::SubpassDependency(
	//		1,
	//		2,
	//		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	//		VK_ACCESS_SHADER_READ_BIT,
	//		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
	//		VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
	//	),
	//	// Second reads the data and works with it
	//	VulkanInitializers::SubpassDependency(
	//		1,
	//		VK_SUBPASS_EXTERNAL,
	//		VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
	//		VK_ACCESS_MEMORY_READ_BIT,
	//		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
	//		VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
	//	),
	//};

	VkRenderPassCreateInfo render_pass_info = VulkanInitializers::RenderPassCreateInfo(attachments, subpasses_descriptions.data(), subpasses_descriptions.size(), subpass_dependency.data(), subpass_dependency.size());

	ErrorCheck(vkCreateRenderPass(
		*m_device->GetVulkanDevice(),
		&render_pass_info,
		nullptr,
		&m_render_pass
	));
	assert(!HasError() && "Unable to initialize render pass");
}

void Renderer::Vulkan::VulkanRenderPass::DeInitRenderPass()
{
	for (int i = 0; i < m_attachments.size(); i++)
	{
		vkDestroyImageView(
			*m_device->GetVulkanDevice(),
			m_attachments[i].color1.view,
			nullptr
		);
		vkDestroyImage(
			*m_device->GetVulkanDevice(),
			m_attachments[i].color1.image,
			nullptr
		);
		vkFreeMemory(
			*m_device->GetVulkanDevice(),
			m_attachments[i].color1.memory,
			nullptr
		);
		vkDestroySampler(
			*m_device->GetVulkanDevice(),
			m_attachments[i].color1.sampler,
			nullptr
		);
		vkDestroyImageView(
			*m_device->GetVulkanDevice(),
			m_attachments[i].color2.view,
			nullptr
		);
		vkDestroyImage(
			*m_device->GetVulkanDevice(),
			m_attachments[i].color2.image,
			nullptr
		);
		vkFreeMemory(
			*m_device->GetVulkanDevice(),
			m_attachments[i].color2.memory,
			nullptr
		);
		vkDestroySampler(
			*m_device->GetVulkanDevice(),
			m_attachments[i].color2.sampler,
			nullptr
		);
		vkDestroyImageView(
			*m_device->GetVulkanDevice(),
			m_attachments[i].depth.view,
			nullptr
		);
		vkDestroyImage(
			*m_device->GetVulkanDevice(),
			m_attachments[i].depth.image,
			nullptr
		);
		vkFreeMemory(
			*m_device->GetVulkanDevice(),
			m_attachments[i].depth.memory,
			nullptr
		);
		vkDestroySampler(
			*m_device->GetVulkanDevice(),
			m_attachments[i].depth.sampler,
			nullptr
		);
	}
	m_attachments.resize(0);




	vkDestroyRenderPass(
		*m_device->GetVulkanDevice(),
		m_render_pass,
		nullptr);
}

void Renderer::Vulkan::VulkanRenderPass::InitFrameBuffer()
{
	m_swap_chain_framebuffers.resize(m_swapchain->GetSwapchainImages().size());
	for (uint32_t i = 0; i < m_swap_chain_framebuffers.size(); i++)
	{
		std::vector<VkImageView> attachments = {
			m_swapchain->GetSwpachainImageViews()[i],
			m_attachments[i].color1.view,
			m_attachments[i].color2.view,
			m_attachments[i].depth.view,
		};
		// Frame buffer create info
		VkFramebufferCreateInfo framebuffer_info = VulkanInitializers::FramebufferCreateInfo(m_swapchain->GetSwapchainExtent(), attachments, m_render_pass);

		ErrorCheck(vkCreateFramebuffer(
			*m_device->GetVulkanDevice(),
			&framebuffer_info,
			nullptr,
			&m_swap_chain_framebuffers[i]
		));
		assert(!HasError() && "Unable to create frame buffer");
	}
}

void Renderer::Vulkan::VulkanRenderPass::DeInitFrameBuffer()
{
	for (auto framebuffer : m_swap_chain_framebuffers)
	{
		vkDestroyFramebuffer(
			*m_device->GetVulkanDevice(),
			framebuffer,
			nullptr
		);
	}
}

void Renderer::Vulkan::VulkanRenderPass::InitCommandBuffers()
{
	if ((m_instance->GetFlags() & VulkanFlags::ActiveCMDRebuild) == VulkanFlags::ActiveCMDRebuild)
	{
		// We will only be using one command buffer as we will rebuild it every frame
		m_command_buffers.resize(1);
	}
	else
	{
		// Resize the command buffers so that there is one for each frame buffer
		m_command_buffers.resize(m_swap_chain_framebuffers.size());
	}
	// Setup the allocation info
	VkCommandBufferAllocateInfo alloc_info = VulkanInitializers::CommandBufferAllocateInfo(
		*m_device->GetGraphicsCommandPool(),
		static_cast<uint32_t>(m_command_buffers.size())
	);
	// Allocate the buffers
	ErrorCheck(vkAllocateCommandBuffers(
		*m_device->GetVulkanDevice(),
		&alloc_info,
		m_command_buffers.data()
	));
	assert(!HasError() && "Unable to allocate command buffers");
}

void Renderer::Vulkan::VulkanRenderPass::InitFences()
{
	for (int i = 0; i < 3; i++)
	{
		VkFence fence;
		VkFenceCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		vkCreateFence(*m_device->GetVulkanDevice(), &info, nullptr, &fence);
		m_fence.push_back(fence);
	}
}

void Renderer::Vulkan::VulkanRenderPass::CreateAttachmentImages(VkFormat format, VkImageUsageFlags usage, FrameBufferAttachment & attachment)
{
	attachment.format = format;

	VkImageAspectFlags aspectMask = 0;
	VkImageLayout imageLayout;

	if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
	{
		aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	}
	if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
	{
		aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	}



	VulkanCommon::CreateImage(m_device, m_swapchain->GetSwapchainExtent(), format, VK_IMAGE_TILING_OPTIMAL, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, attachment.image, attachment.memory);
	VulkanCommon::CreateImageView(m_device, attachment.image, format, aspectMask, attachment.view);


	{ // Define image sampler
		
		VkSamplerCreateInfo sampler_info = VulkanInitializers::SamplerCreateInfo();
		sampler_info.magFilter = VK_FILTER_LINEAR;
		sampler_info.minFilter = VK_FILTER_LINEAR;
		sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		sampler_info.mipLodBias = 0.0f;
		sampler_info.compareOp = VK_COMPARE_OP_NEVER;
		sampler_info.minLod = 0.0f;
		// Set max level-of-detail to mip level count of the texture
		sampler_info.maxLod = (float)1;
		// Enable anisotropic filtering
		// This feature is optional, so we must check if it's supported on the device
		if (m_device->GetVulkanPhysicalDevice()->GetDeviceFeatures()->samplerAnisotropy)
		{
			// Use max. level of anisotropy for this example
			sampler_info.maxAnisotropy = m_device->GetVulkanPhysicalDevice()->GetPhysicalDeviceProperties()->limits.maxSamplerAnisotropy;
			sampler_info.anisotropyEnable = VK_TRUE;
		}
		else
		{
			// The device does not support anisotropic filtering
			sampler_info.maxAnisotropy = 1.0;
			sampler_info.anisotropyEnable = VK_FALSE;
		}
		sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		ErrorCheck(vkCreateSampler(
			*m_device->GetVulkanDevice(),
			&sampler_info,
			nullptr,
			&attachment.sampler
		));
	}
	



	if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
		VulkanCommon::TransitionImageLayout(m_device, attachment.image, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });
}

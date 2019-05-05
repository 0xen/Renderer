#include <renderer/vulkan/VulkanAcceleration.hpp>
#include <renderer/vulkan/VulkanBuffer.hpp>
#include <renderer/vulkan/VulkanCommon.hpp>
#include <renderer/vulkan/VulkanDevice.hpp>
#include <renderer/vulkan/VulkanModelPool.hpp>
#include <renderer/vulkan/VulkanInitializers.hpp>
#include <renderer/vulkan/VulkanVertexBuffer.hpp>
#include <renderer/vulkan/VulkanIndexBuffer.hpp>
#include <renderer\vulkan\VulkanBufferData.hpp>


Renderer::Vulkan::VulkanAcceleration::VulkanAcceleration(VulkanDevice* device)
{
	m_device = device;

}

Renderer::Vulkan::VulkanAcceleration::~VulkanAcceleration()
{
}

void Renderer::Vulkan::VulkanAcceleration::AttachModelPool(VulkanModelPool * pool)
{
	m_model_pools.push_back(pool);
}

void Renderer::Vulkan::VulkanAcceleration::Build()
{
	VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
	m_device->GetGraphicsCommand(&commandBuffer, true);

	m_bottom_level_as.resize(m_model_pools.size()); // Will be amount of geometry instances

	std::vector<std::pair<VkAccelerationStructureNV, glm::mat4x4>> instances;


	for (int i = 0; i < m_model_pools.size(); i++)
	{
		m_bottom_level_as[i] = CreateBottomLevelAS(commandBuffer, m_model_pools[i]);

		glm::mat4x4 mat = glm::mat4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
		instances.push_back({ m_bottom_level_as[i].structure, mat });
	}


	CreateTopLevelAS(commandBuffer, instances);

	vkEndCommandBuffer(commandBuffer);

	m_device->SubmitGraphicsCommand(&commandBuffer, 1);
	m_device->FreeGraphicsCommand(&commandBuffer, 1);
}

VkWriteDescriptorSetAccelerationStructureNV Renderer::Vulkan::VulkanAcceleration::GetDescriptorAcceleration()
{
	return VulkanInitializers::WriteDescriptorSetAccelerator(m_top_level_as.structure);
}

Renderer::Vulkan::AccelerationStructure Renderer::Vulkan::VulkanAcceleration::CreateBottomLevelAS(VkCommandBuffer commandBuffer, VulkanModelPool * pool)
{

	std::vector<VkGeometryNV> vertex_buffers = {};
	AccelerationStructure acceleration;

	if (pool->Indexed())
	{
		VulkanVertexBuffer* vertexBuffer = static_cast<VulkanVertexBuffer*>(pool->GetVertexBuffer());
		VulkanIndexBuffer* indexBuffer = static_cast<VulkanIndexBuffer*>(pool->GetIndexBuffer());
		VkGeometryNV geom = VulkanInitializers::CreateRayTraceGeometry(vertexBuffer->GetBufferData(BufferSlot::Primary)->buffer, 0, vertexBuffer->GetElementCount(BufferSlot::Primary),
			vertexBuffer->GetIndexSize(BufferSlot::Primary), indexBuffer->GetBufferData(BufferSlot::Primary)->buffer, 0, indexBuffer->GetElementCount(BufferSlot::Primary), VK_NULL_HANDLE, 0, true);
		vertex_buffers.push_back(geom);
	}
	else
	{
		VulkanVertexBuffer* vertexBuffer = static_cast<VulkanVertexBuffer*>(pool->GetVertexBuffer());
		VkGeometryNV geom = VulkanInitializers::CreateRayTraceGeometry(vertexBuffer->GetBufferData(BufferSlot::Primary)->buffer, 0, vertexBuffer->GetElementCount(BufferSlot::Primary),
			vertexBuffer->GetIndexSize(BufferSlot::Primary), nullptr, 0, 0, VK_NULL_HANDLE, 0, true);
		vertex_buffers.push_back(geom);
	}

	VkBuildAccelerationStructureFlagsNV flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_NV;

	VkAccelerationStructureNV acceleration_structure;
	{
		VkAccelerationStructureInfoNV acceleration_structure_info = VulkanInitializers::AccelerationStructureInfoNV(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV,flags, vertex_buffers.data(), vertex_buffers.size(), 0);

		VkAccelerationStructureCreateInfoNV create_info = VulkanInitializers::AccelerationStructureCreateInfoNV(acceleration_structure_info);


		VkResult code = vkCreateAccelerationStructureNV(*m_device->GetVulkanDevice(), &create_info, nullptr, &acceleration_structure);

		// Once the overall size of the geometry is known, we can create the handle
		// for the acceleration structure
		acceleration.structure = acceleration_structure;
	}
	
	// The AS needs some space to store temporary info, this spaec requirment is dependent on the scene complexity
	VkDeviceSize scratch_size = 0;

	// We need to calculate the final AS size
	VkDeviceSize result_size = 0;

	VkAccelerationStructureMemoryRequirementsInfoNV memory_requirments_info = VulkanInitializers::AccelerationStructureMemoryRequirmentsInfoNV(acceleration_structure);
	
	VkMemoryRequirements2 memoryRequirements;
	
	{
		vkGetAccelerationStructureMemoryRequirementsNV(*m_device->GetVulkanDevice(), &memory_requirments_info, &memoryRequirements);

		// Size of the resulting AS
		result_size = memoryRequirements.memoryRequirements.size;
	}

	{
		// Store the memory requirments
		memory_requirments_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
		vkGetAccelerationStructureMemoryRequirementsNV(*m_device->GetVulkanDevice(), &memory_requirments_info, &memoryRequirements);

		scratch_size = memoryRequirements.memoryRequirements.size;
	}
	
	{
		memory_requirments_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_NV;
		vkGetAccelerationStructureMemoryRequirementsNV(*m_device->GetVulkanDevice(), &memory_requirments_info, &memoryRequirements);

		scratch_size = scratch_size > memoryRequirements.memoryRequirements.size ? scratch_size : memoryRequirements.memoryRequirements.size;

	}

	{
		VulkanCommon::CreateBuffer(
			m_device,
			scratch_size,
			VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			acceleration.scratch
		);
		//VulkanCommon::MapBufferMemory(m_device, acceleration.scratch, acceleration.scratch.size);
	}



	{
		VulkanCommon::CreateBuffer(
			m_device,
			result_size,
			VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			acceleration.result
		);
		//VulkanCommon::MapBufferMemory(m_device, acceleration.result, acceleration.result.size);
	}





	// Generate the bottom level AS
	{

		VkBindAccelerationStructureMemoryInfoNV acceleration_memory_info = VulkanInitializers::AccelerationStructureMemoryInfoNV(acceleration_structure, acceleration.result.buffer_memory);


		VkResult code = vkBindAccelerationStructureMemoryNV(*m_device->GetVulkanDevice(), 1, &acceleration_memory_info);


		VkAccelerationStructureInfoNV acceleration_structure_info = VulkanInitializers::AccelerationStructureInfo(0, vertex_buffers);


		vkCmdBuildAccelerationStructureNV(commandBuffer, &acceleration_structure_info, VK_NULL_HANDLE, 0, VK_FALSE,
			acceleration_structure, VK_NULL_HANDLE , acceleration.scratch.buffer, 0);


		VkMemoryBarrier memoryBarrier;
		memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		memoryBarrier.pNext = nullptr;
		memoryBarrier.srcAccessMask =
			VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
		memoryBarrier.dstAccessMask =
			VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;

		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
			VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV, 0, 1, &memoryBarrier,
			0, nullptr, 0, nullptr);

	}

	return acceleration;
}

void Renderer::Vulkan::VulkanAcceleration::CreateTopLevelAS(VkCommandBuffer commandBuffer, std::vector<std::pair<VkAccelerationStructureNV, glm::mat4x4>>& instances)
{

	for (uint32_t i = 0; i < instances.size(); i++)
	{
		// we set the hit group to 2 * i as there are two types of rays being used in this example, shadow rays and camera rays.
		m_as_instance.push_back({ instances[i].first,instances[i].second, i, i * 2 });
	}

	VkBuildAccelerationStructureFlagsNV flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_NV;

	VkAccelerationStructureNV acceleration_structure;
	{
		VkAccelerationStructureInfoNV acceleration_structure_info = VulkanInitializers::AccelerationStructureInfoNV(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV, flags, VK_NULL_HANDLE, 0, instances.size());

		VkAccelerationStructureCreateInfoNV create_info = VulkanInitializers::AccelerationStructureCreateInfoNV(acceleration_structure_info);


		VkResult code = vkCreateAccelerationStructureNV(*m_device->GetVulkanDevice(), &create_info, nullptr, &acceleration_structure);

		// Once the overall size of the geometry is known, we can create the handle
		// for the acceleration structure
		m_top_level_as.structure = acceleration_structure;
	}




	// The AS needs some space to store temporary info, this spaec requirment is dependent on the scene complexity
	VkDeviceSize scratch_size = 0;

	// We need to calculate the final AS size
	VkDeviceSize result_size = 0;

	// Required GPU memory to store instance
	VkDeviceSize instances_size = 0;


	VkAccelerationStructureMemoryRequirementsInfoNV memory_requirments_info = VulkanInitializers::AccelerationStructureMemoryRequirmentsInfoNV(acceleration_structure);

	VkMemoryRequirements2 memoryRequirements;

	{
		vkGetAccelerationStructureMemoryRequirementsNV(*m_device->GetVulkanDevice(), &memory_requirments_info, &memoryRequirements);

		// Size of the resulting AS
		result_size = memoryRequirements.memoryRequirements.size;
	}

	{
		// Store the memory requirments
		memory_requirments_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
		vkGetAccelerationStructureMemoryRequirementsNV(*m_device->GetVulkanDevice(), &memory_requirments_info, &memoryRequirements);

		scratch_size = memoryRequirements.memoryRequirements.size;
	}

	{
		memory_requirments_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_NV;
		vkGetAccelerationStructureMemoryRequirementsNV(*m_device->GetVulkanDevice(), &memory_requirments_info, &memoryRequirements);

		scratch_size = scratch_size > memoryRequirements.memoryRequirements.size ? scratch_size : memoryRequirements.memoryRequirements.size;

	}


	instances_size = instances.size() * sizeof(VkGeometryInstance);




	{
		VulkanCommon::CreateBuffer(
			m_device,
			scratch_size,
			VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			m_top_level_as.scratch
		);
		//VulkanCommon::MapBufferMemory(m_device, acceleration.scratch, acceleration.scratch.size);
	}



	{
		VulkanCommon::CreateBuffer(
			m_device,
			result_size,
			VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			m_top_level_as.result
		);
		//VulkanCommon::MapBufferMemory(m_device, acceleration.result, acceleration.result.size);
	}



	{
		// this buffer is used to describe a instances: ID, shader binding and matricies
		VulkanCommon::CreateBuffer(
			m_device,
			instances_size,
			VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			m_top_level_as.instances
		);
		//VulkanCommon::MapBufferMemory(m_device, acceleration.result, acceleration.result.size);
	}




	// Generate top level
	std::vector<VkGeometryInstance> geometryInstances;


	for (const auto& instance : m_as_instance)
	{
		uint64_t accelerationStructureHandle = 0;
		VkResult code = vkGetAccelerationStructureHandleNV(*m_device->GetVulkanDevice(), instance.bottomLevelAS, sizeof(uint64_t),
			&accelerationStructureHandle);
		if (code != VK_SUCCESS)
		{
			throw std::logic_error("vkGetAccelerationStructureHandleNV failed");
		}

		VkGeometryInstance gInst;
		glm::mat4x4        transp = glm::transpose(instance.transform);
		memcpy(gInst.transform, &transp, sizeof(gInst.transform));
		gInst.instanceId = instance.instanceID;
		// The visibility mask is always set of 0xFF, but if some instances would need to be ignored in
		// some cases, this flag should be passed by the application
		gInst.mask = 0xff;
		// Set the hit group index, that will be used to find the shader code to execute when hitting
		// the geometry
		gInst.instanceOffset = instance.hitGroupIndex;
		// Disable culling - more fine control could be provided by the application
		gInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;
		gInst.accelerationStructureHandle = accelerationStructureHandle;
		geometryInstances.push_back(gInst);
	}



	// Copy the instance descriptors into the provided mappable buffer.



	VkDeviceSize instancesBufferSize = geometryInstances.size() * sizeof(VkGeometryInstance);
	VulkanCommon::MapBufferMemory(m_device, m_top_level_as.instances, instancesBufferSize);


	memcpy(m_top_level_as.instances.mapped_memory, geometryInstances.data(), instancesBufferSize);

	VulkanCommon::UnMapBufferMemory(m_device, m_top_level_as.instances);


	VkBindAccelerationStructureMemoryInfoNV bindInfo = VulkanInitializers::AccelerationStructureMemoryInfoNV(m_top_level_as.structure, m_top_level_as.result.buffer_memory);


	VkResult code = vkBindAccelerationStructureMemoryNV(*m_device->GetVulkanDevice(), 1, &bindInfo);


	VkAccelerationStructureInfoNV acceleration_structure_info = VulkanInitializers::AccelerationStructureInfo(0, geometryInstances.size());


	vkCmdBuildAccelerationStructureNV(commandBuffer, &acceleration_structure_info, m_top_level_as.instances.buffer, 0, VK_FALSE,
		acceleration_structure, VK_NULL_HANDLE, m_top_level_as.scratch.buffer, 0);

	// Ensure that the build will be finished before using the AS using a barrier
	VkMemoryBarrier memoryBarrier;
	memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	memoryBarrier.pNext = nullptr;
	memoryBarrier.srcAccessMask =
		VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
	memoryBarrier.dstAccessMask =
		VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;

	vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
		VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV, 0, 1, &memoryBarrier,
		0, nullptr, 0, nullptr);

}

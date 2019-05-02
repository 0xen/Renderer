#pragma once

#include <renderer\vulkan\VulkanHeader.hpp>
#include <renderer\vulkan\VulkanBufferData.hpp>

#include <glm\glm.hpp>

#include <vector>

namespace Renderer
{
	namespace Vulkan
	{
		struct AccelerationStructure
		{
			VulkanBufferData scratch;
			VulkanBufferData result;
			VulkanBufferData instances;

			/*VkBuffer                  scratchBuffer = VK_NULL_HANDLE;
			VkDeviceMemory            scratchMem = VK_NULL_HANDLE;
			VkBuffer                  resultBuffer = VK_NULL_HANDLE;
			VkDeviceMemory            resultMem = VK_NULL_HANDLE;
			VkBuffer                  instancesBuffer = VK_NULL_HANDLE;
			VkDeviceMemory            instancesMem = VK_NULL_HANDLE;*/
			VkAccelerationStructureNV structure = VK_NULL_HANDLE;
		};

		struct ASInstance
		{
			VkAccelerationStructureNV bottomLevelAS;

			const glm::mat4x4 transform;

			// Instance Id that is visible to the shader
			uint32_t instanceID;

			// Hit group index used to fetch the shaders from the SBT
			uint32_t hitGroupIndex;
		};

		struct VkGeometryInstance
		{
			/// Transform matrix, containing only the top 3 rows
			float transform[12];
			/// Instance index
			uint32_t instanceId : 24;
			/// Visibility mask
			uint32_t mask : 8;
			/// Index of the hit group which will be invoked when a ray hits the instance
			uint32_t instanceOffset : 24;
			/// Instance flags, such as culling
			uint32_t flags : 8;
			/// Opaque handle of the bottom-level acceleration structure
			uint64_t accelerationStructureHandle;
		};

		static_assert(sizeof(VkGeometryInstance) == 64,
			"VkGeometryInstance structure compiles to incorrect size");

		class VulkanDevice;
		class VulkanModelPool;

		class VulkanAcceleration
		{
		public:
			VulkanAcceleration(VulkanDevice* device);
			~VulkanAcceleration();

			void AttachModelPool(VulkanModelPool* pool);

			void Build();

			VkWriteDescriptorSetAccelerationStructureNV GetDescriptorAcceleration();

		private:
			AccelerationStructure CreateBottomLevelAS(VkCommandBuffer commandBuffer, VulkanModelPool* pool);
			void CreateTopLevelAS(VkCommandBuffer commandBuffer, std::vector<std::pair<VkAccelerationStructureNV, glm::mat4x4>>& instances);

			VulkanDevice* m_device;

			AccelerationStructure m_top_level_as;
			std::vector<ASInstance> m_as_instance;
			std::vector<AccelerationStructure> m_bottom_level_as;
			std::vector<VulkanModelPool*> m_model_pools;

		};

	}
}
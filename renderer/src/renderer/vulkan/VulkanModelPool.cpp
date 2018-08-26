#include <renderer/vulkan/VulkanModelPool.hpp>
#include <renderer/vulkan/VulkanUniformBuffer.hpp>
#include <renderer/vulkan/VulkanModel.hpp>
#include <renderer/vulkan/VulkanVertexBuffer.hpp>
#include <renderer/vulkan/VulkanIndexBuffer.hpp>
#include <renderer/vulkan/VulkanDescriptorSet.hpp>
#include <renderer/vulkan/VulkanPipeline.hpp>


Renderer::Vulkan::VulkanModelPool::VulkanModelPool(VulkanDevice* device, IVertexBuffer * vertex_buffer, IIndexBuffer * index_buffer) :
	IModelPool(vertex_buffer, index_buffer)
{
	m_device = device;
	m_current_index = 0;
	m_change = false;

	m_indirect_command.instanceCount = 0;
	m_indirect_command.firstInstance = 0;
	m_indirect_command.vertexOffset = 0;
	m_indirect_command.firstIndex = 0;
	m_indirect_command.indexCount = index_buffer->GetElementCount();

	m_indirect_draw_buffer = new VulkanBuffer(device, &m_indirect_command, sizeof(m_indirect_command), 1
		, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	m_indirect_draw_buffer->SetData();
}

Renderer::Vulkan::VulkanModelPool::~VulkanModelPool()
{

}

Renderer::IModel * Renderer::Vulkan::VulkanModelPool::CreateModel()
{
	VulkanModel* model = new VulkanModel(m_current_index);
	m_models[m_current_index] = model;
	for (auto buffer = m_buffers.begin(); buffer != m_buffers.end(); buffer++)
	{
		void* data = buffer->second->GetDataPointer();
		model->SetDataPointer(buffer->first, ((char*)data) + (buffer->second->GetIndexSize() * m_current_index));
	}
	m_change = true;
	m_current_index++;
	// Update how many models we need to render
	m_indirect_command.instanceCount = m_current_index;
	m_indirect_draw_buffer->SetData();

	return model;
}

void Renderer::Vulkan::VulkanModelPool::AttachBuffer(unsigned int index, IUniformBuffer * buffer)
{
	m_buffers[index] = dynamic_cast<VulkanUniformBuffer*>(buffer);
}

void Renderer::Vulkan::VulkanModelPool::AttachDescriptorSet(unsigned int index, IDescriptorSet * descriptor_set)
{
	m_descriptor_sets[index] = static_cast<VulkanDescriptorSet*>(descriptor_set);
}

void Renderer::Vulkan::VulkanModelPool::AttachToCommandBuffer(VkCommandBuffer & command_buffer, VulkanPipeline* pipeline)
{
	VkDeviceSize offsets[] = { 0 };
	for(auto it = m_descriptor_sets.begin(); it!= m_descriptor_sets.end(); it++)
	{
		vkCmdBindDescriptorSets(
			command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			pipeline->GetPipelineLayout(),
			it->first,
			1,
			&it->second->GetDescriptorSet(),
			0,
			NULL
		);
	}

	vkCmdBindVertexBuffers(
		command_buffer,
		0,
		1,
		&dynamic_cast<VulkanVertexBuffer*>(m_vertex_buffer)->GetBufferData()->buffer,
		offsets
	);

	vkCmdBindIndexBuffer(
		command_buffer,
		dynamic_cast<VulkanIndexBuffer*>(m_index_buffer)->GetBufferData()->buffer,
		0,
		VK_INDEX_TYPE_UINT16
	);

	std::vector<VkBuffer> vertex_buffers;

	for (auto buffer  = m_buffers.begin(); buffer!= m_buffers.end();buffer++)
	{
		vertex_buffers.push_back(buffer->second->GetBufferData()->buffer);
	}

	vkCmdBindVertexBuffers(
		command_buffer,
		1,
		vertex_buffers.size(),
		vertex_buffers.data(),
		offsets
	);

	vkCmdDrawIndexedIndirect(
		command_buffer,
		m_indirect_draw_buffer->GetBufferData()->buffer,
		0,
		1,
		sizeof(VkDrawIndexedIndirectCommand)
	);
}

bool Renderer::Vulkan::VulkanModelPool::HasChanged()
{
	if (m_change)
	{
		m_change = false;
		return true;
	}
	return false;
}

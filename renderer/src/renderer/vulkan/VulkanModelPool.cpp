#include <renderer/vulkan/VulkanModelPool.hpp>
#include <renderer/vulkan/VulkanUniformBuffer.hpp>
#include <renderer/vulkan/VulkanModel.hpp>
#include <renderer/vulkan/VulkanVertexBuffer.hpp>
#include <renderer/vulkan/VulkanIndexBuffer.hpp>
#include <renderer/vulkan/VulkanDescriptorSet.hpp>
#include <renderer/vulkan/VulkanPipeline.hpp>
#include <renderer/vulkan/VulkanPhysicalDevice.hpp>



const unsigned int Renderer::Vulkan::VulkanModelPool::m_indirect_array_padding = 100;

Renderer::Vulkan::VulkanModelPool::VulkanModelPool(VulkanDevice * device, IVertexBuffer * vertex_buffer) :
	IModelPool(vertex_buffer)
{
	m_device = device;
	m_current_index = 0;
	m_vertex_draw_count = vertex_buffer->GetElementCount(BufferSlot::Primary);
	m_change = false;

	ResizeIndirectArray(m_indirect_array_padding);
}

Renderer::Vulkan::VulkanModelPool::VulkanModelPool(VulkanDevice* device, IVertexBuffer * vertex_buffer, IIndexBuffer * index_buffer) :
	IModelPool(vertex_buffer, index_buffer)
{
	m_device = device;
	m_current_index = 0;
	m_vertex_draw_count = index_buffer->GetElementCount(BufferSlot::Primary);
	m_change = false;

	ResizeIndirectArray(m_indirect_array_padding);
}

Renderer::Vulkan::VulkanModelPool::~VulkanModelPool()
{
	delete m_indirect_draw_buffer;
}

Renderer::IModel * Renderer::Vulkan::VulkanModelPool::CreateModel()
{
	VulkanModel* model = new VulkanModel(this, m_current_index);
	m_models[m_current_index] = model;
	for (auto buffer = m_buffers.begin(); buffer != m_buffers.end(); buffer++)
	{
		void* data = buffer->second->GetDataPointer(BufferSlot::Primary);
		model->SetDataPointer(buffer->first, ((char*)data) + (buffer->second->GetIndexSize(BufferSlot::Primary) * m_current_index));
	}
	m_change = true;

	Render(m_current_index, true);

	m_current_index++;

	return model;
}

void Renderer::Vulkan::VulkanModelPool::Update()
{
	for (auto it = m_buffers.begin(); it != m_buffers.end(); it++)
	{
		it->second->SetData(BufferSlot::Primary);
	}
}

void Renderer::Vulkan::VulkanModelPool::AttachBuffer(unsigned int index, IUniformBuffer * buffer)
{
	m_buffers[index] = dynamic_cast<VulkanUniformBuffer*>(buffer);
}

void Renderer::Vulkan::VulkanModelPool::AttachDescriptorSet(unsigned int index, IDescriptorSet * descriptor_set)
{
	m_descriptor_sets[index] = static_cast<VulkanDescriptorSet*>(descriptor_set);
}

void Renderer::Vulkan::VulkanModelPool::SetVertexDrawCount(unsigned int count)
{
	m_vertex_draw_count = count;


	if (Indexed())
	{
		for (int i = 0; i < m_indexed_indirect_command.size(); i++)
		{
			VkDrawIndexedIndirectCommand& indexed_indirect_command = m_indexed_indirect_command[i];
			indexed_indirect_command.indexCount = m_vertex_draw_count;
		}
	}
	else
	{
		for (int i = 0; i < m_vertex_indirect_command.size(); i++)
		{
			VkDrawIndirectCommand& vertex_indirect_command = m_vertex_indirect_command[i];
			vertex_indirect_command.vertexCount = m_vertex_draw_count;
		}
	}
	m_indirect_draw_buffer->SetData(BufferSlot::Primary);
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
		&dynamic_cast<VulkanVertexBuffer*>(m_vertex_buffer)->GetBufferData(BufferSlot::Primary)->buffer,
		offsets
	);

	if (Indexed())
	{
		vkCmdBindIndexBuffer(
			command_buffer,
			dynamic_cast<VulkanIndexBuffer*>(m_index_buffer)->GetBufferData(BufferSlot::Primary)->buffer,
			0,
			VK_INDEX_TYPE_UINT16
		);
	}


	if (m_buffers.size() > 0)
	{
		std::vector<VkBuffer> vertex_buffers;
		for (auto buffer = m_buffers.begin(); buffer != m_buffers.end(); buffer++)
		{
			vertex_buffers.push_back(buffer->second->GetBufferData(BufferSlot::Primary)->buffer);
		}
		vkCmdBindVertexBuffers(
			command_buffer,
			1,
			vertex_buffers.size(),
			vertex_buffers.data(),
			offsets
		);
	}
	

	// Check to see if we can render all models in one draw pass
	if (m_device->GetVulkanPhysicalDevice()->GetDeviceFeatures()->multiDrawIndirect &&
		m_device->GetVulkanPhysicalDevice()->GetPhysicalDeviceProperties()->limits.maxDrawIndirectCount >= m_current_index)
	{
		// Render using a index buffer if one was provided
		if (Indexed())
		{
			vkCmdDrawIndexedIndirect(
				command_buffer,
				m_indirect_draw_buffer->GetBufferData(BufferSlot::Primary)->buffer,
				0,
				m_current_index,
				sizeof(VkDrawIndexedIndirectCommand)
			);
		}
		else
		{
			vkCmdDrawIndirect(
				command_buffer,
				m_indirect_draw_buffer->GetBufferData(BufferSlot::Primary)->buffer,
				0,
				m_current_index,
				sizeof(VkDrawIndirectCommand)
			);
		}
	}
	else // If we cant, loop through for each draw*/
	{
		if (Indexed())
		{
			for (auto j = 0; j < m_current_index; j++)
			{
				vkCmdDrawIndexedIndirect(
					command_buffer,
					m_indirect_draw_buffer->GetBufferData(BufferSlot::Primary)->buffer,
					j * sizeof(VkDrawIndexedIndirectCommand), 
					1, 
					sizeof(VkDrawIndexedIndirectCommand));
			}
		}
		else
		{
			for (auto j = 0; j < m_current_index; j++)
			{
				vkCmdDrawIndirect(
					command_buffer,
					m_indirect_draw_buffer->GetBufferData(BufferSlot::Primary)->buffer,
					j * sizeof(VkDrawIndirectCommand),
					1,
					sizeof(VkDrawIndirectCommand)
				);
			}
		}
	}

	
	
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

void Renderer::Vulkan::VulkanModelPool::ResizeIndirectArray(unsigned int size)
{

	unsigned int old_size;
	unsigned int instance_size;
	if (Indexed())
	{
		instance_size = sizeof(VkDrawIndexedIndirectCommand);
		old_size = m_indexed_indirect_command.size();
		m_indexed_indirect_command.resize(size);
	}
	else
	{
		instance_size = sizeof(VkDrawIndirectCommand);
		old_size = m_vertex_indirect_command.size();
		m_vertex_indirect_command.resize(size);
	}
	// If the buffer is not created, create it
	if (m_indirect_draw_buffer == nullptr)
	{
		if (Indexed())
		{
			// Set the data for the model pool
			for (int i = 0; i < size; i++)
			{
				VkDrawIndexedIndirectCommand& indexed_indirect_command = m_indexed_indirect_command[i];
				indexed_indirect_command.indexCount = GetIndexBuffer()->GetElementCount(BufferSlot::Primary);
				indexed_indirect_command.instanceCount = 0;
				indexed_indirect_command.firstIndex = 0;
				indexed_indirect_command.vertexOffset = 0;
				indexed_indirect_command.firstInstance = i;
			}
			// Create the vulkan buffer
			m_indirect_draw_buffer = new VulkanBuffer(m_device, BufferChain::Single, m_indexed_indirect_command.data(), instance_size, size,
				VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

			m_indirect_draw_buffer->SetData(BufferSlot::Primary);
		}
		else
		{
			// Set the data for the model pool
			for (int i = 0; i < size; i++)
			{
				VkDrawIndirectCommand& vertex_indirect_command = m_vertex_indirect_command[i];
				vertex_indirect_command.firstInstance = i;
				vertex_indirect_command.firstVertex = 0;
				vertex_indirect_command.instanceCount = 0;
				vertex_indirect_command.vertexCount = GetVertexBuffer()->GetElementCount(BufferSlot::Primary);
			}
			// Create the vulkan buffer
			m_indirect_draw_buffer = new VulkanBuffer(m_device, BufferChain::Single, m_vertex_indirect_command.data(), instance_size, size,
				VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

			m_indirect_draw_buffer->SetData(BufferSlot::Primary);
		}

	}
	else // Vulkan buffer already created
	{
		if (Indexed())
		{
			for (int i = old_size; i < size; i++)
			{
				VkDrawIndexedIndirectCommand& indexed_indirect_command = m_indexed_indirect_command[i];
				indexed_indirect_command.indexCount = GetIndexBuffer()->GetElementCount(BufferSlot::Primary);
				indexed_indirect_command.instanceCount = 0;
				indexed_indirect_command.firstIndex = 0;
				indexed_indirect_command.vertexOffset = 0;
				indexed_indirect_command.firstInstance = i;
			}
			m_indirect_draw_buffer->Resize(BufferSlot::Primary, m_indexed_indirect_command.data(), size);
		}
		else
		{
			for (int i = old_size; i < size; i++)
			{
				VkDrawIndirectCommand& vertex_indirect_command = m_vertex_indirect_command[i];
				vertex_indirect_command.firstInstance = i;
				vertex_indirect_command.firstVertex = 0;
				vertex_indirect_command.instanceCount = 0;
				vertex_indirect_command.vertexCount = GetVertexBuffer()->GetElementCount(BufferSlot::Primary);
			}
			m_indirect_draw_buffer->Resize(BufferSlot::Primary, m_vertex_indirect_command.data(), size);
		}
		m_indirect_draw_buffer->SetData(BufferSlot::Primary);
	}


	/*
	unsigned int instance_size;
	void* indirect_command;

	if (Indexed())
	{
		instance_size = sizeof(VkDrawIndexedIndirectCommand);
		indirect_command = new VkDrawIndexedIndirectCommand[size];
	}
	else
	{
		instance_size = sizeof(VkDrawIndirectCommand);
		indirect_command = new VkDrawIndirectCommand[size];
	}

	// Copy data from old array to new array
	if (m_indirect_command != nullptr)
	{
		if (size < m_indirect_draw_buffer->GetElementCount(BufferSlot::Primary))
		{
			// Unable to currently handle this
			assert(0 && "Currently unable to decrease indirect draw buffer size");
		}
		memcpy(indirect_command, m_indirect_command, m_indirect_draw_buffer->GetElementCount(BufferSlot::Primary) * m_indirect_draw_buffer->GetIndexSize(BufferSlot::Primary));

		delete m_indirect_command;
		m_indirect_command = indirect_command;


		for (int i = m_indirect_draw_buffer->GetElementCount(BufferSlot::Primary); i < size; i++)
		{
			if (Indexed())
			{
				VkDrawIndexedIndirectCommand& indexed_indirect_command = m_indexed_indirect_command[i];
				indexed_indirect_command.indexCount = m_vertex_draw_count;
				indexed_indirect_command.instanceCount = 0;
				indexed_indirect_command.firstIndex = 0;
				indexed_indirect_command.vertexOffset = 0;
				indexed_indirect_command.firstInstance = i;
			}
			else
			{
				VkDrawIndirectCommand& vertex_indirect_command = m_vertex_indirect_command[i];
				vertex_indirect_command.firstInstance = i;
				vertex_indirect_command.firstVertex = 0;
				vertex_indirect_command.instanceCount = 0;
				vertex_indirect_command.vertexCount = m_vertex_draw_count;
			}
		}


		m_indirect_draw_buffer->Resize(BufferSlot::Primary, m_indirect_command, size);


		m_indirect_draw_buffer->SetData(BufferSlot::Primary);
	}
	else // Create new array and buffer
	{
		m_indirect_command = indirect_command;
		m_indirect_draw_buffer = new VulkanBuffer(m_device, BufferChain::Single, m_indirect_command, instance_size, size,
			VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		for (int i = 0; i < size; i++)
		{
			if (Indexed())
			{
				VkDrawIndexedIndirectCommand& indexed_indirect_command = m_indexed_indirect_command[i];
				indexed_indirect_command.indexCount = GetIndexBuffer()->GetElementCount(BufferSlot::Primary);
				indexed_indirect_command.instanceCount = 0;
				indexed_indirect_command.firstIndex = 0;
				indexed_indirect_command.vertexOffset = 0;
				indexed_indirect_command.firstInstance = i;
			}
			else
			{
				VkDrawIndirectCommand& vertex_indirect_command = m_vertex_indirect_command[i];
				vertex_indirect_command.firstInstance = i;
				vertex_indirect_command.firstVertex = 0;
				vertex_indirect_command.instanceCount = 0;
				vertex_indirect_command.vertexCount = GetVertexBuffer()->GetElementCount(BufferSlot::Primary);
			}
		}
		m_indirect_draw_buffer->SetData(BufferSlot::Primary);
	}*/
}

void Renderer::Vulkan::VulkanModelPool::Render(unsigned int index, bool should_render)
{
	if (index + 1 >= m_indirect_draw_buffer->GetElementCount(BufferSlot::Primary))
	{
		ResizeIndirectArray(index + m_indirect_array_padding);
	}

	if (Indexed())
	{
		VkDrawIndexedIndirectCommand& indexed_indirect_command = m_indexed_indirect_command[index];
		indexed_indirect_command.instanceCount = (should_render ? 1 : 0);
	}
	else
	{
		VkDrawIndirectCommand& vertex_indirect_command = m_vertex_indirect_command[index];
		vertex_indirect_command.instanceCount = (should_render ? 1 : 0);
	}

	m_indirect_draw_buffer->SetData(BufferSlot::Primary,index, 1);
}


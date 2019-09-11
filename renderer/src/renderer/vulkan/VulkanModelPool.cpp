#include <renderer/vulkan/VulkanModelPool.hpp>
#include <renderer/vulkan/VulkanBufferPool.hpp>
#include <renderer/vulkan/VulkanUniformBuffer.hpp>
#include <renderer/vulkan/VulkanModel.hpp>
#include <renderer/vulkan/VulkanVertexBuffer.hpp>
#include <renderer/vulkan/VulkanIndexBuffer.hpp>
#include <renderer/vulkan/VulkanDescriptorSet.hpp>
#include <renderer/vulkan/VulkanGraphicsPipeline.hpp>
#include <renderer/vulkan/VulkanPhysicalDevice.hpp>
#include <renderer/vulkan/VulkanRenderPass.hpp>
#include <renderer/vulkan/VulkanSwapchain.hpp>




const unsigned int Renderer::Vulkan::VulkanModelPool::m_indirect_array_padding = 100;

Renderer::Vulkan::VulkanModelPool::VulkanModelPool(VulkanDevice * device, VulkanVertexBuffer * vertex_buffer, unsigned int vertex_offset, unsigned int vertex_size, ModelPoolUsage usage) :
	m_vertex_offset(vertex_offset), m_vertex_size(vertex_size), m_model_pool_usage(usage)
{
	m_vertex_buffer = vertex_buffer;
	m_device = device;
	m_current_index = 0;
	m_largest_index = 0;
	m_change = false;
	m_indexed = false;
	m_allow_custom_scissors = false;

	ResizeIndirectArray(m_indirect_array_padding);
}

Renderer::Vulkan::VulkanModelPool::VulkanModelPool(VulkanDevice* device, VulkanVertexBuffer * vertex_buffer, unsigned int vertex_offset, unsigned int vertex_size, VulkanIndexBuffer * index_buffer, unsigned int index_offset, unsigned int index_size, ModelPoolUsage usage) :
	m_vertex_offset(vertex_offset), m_vertex_size(vertex_size), m_index_offset(index_offset), m_index_size(index_size), m_model_pool_usage(usage)
{
	m_vertex_buffer = vertex_buffer;
	m_index_buffer = index_buffer;
	m_device = device;
	m_current_index = 0;
	m_largest_index = 0;
	m_change = false;
	m_indexed = true;
	m_allow_custom_scissors = false;

	ResizeIndirectArray(m_indirect_array_padding);
}

Renderer::Vulkan::VulkanModelPool::~VulkanModelPool()
{
	delete m_indirect_draw_buffer;
}

bool Renderer::Vulkan::VulkanModelPool::Indexed()
{
	return m_indexed;
}

Renderer::Vulkan::VulkanModel * Renderer::Vulkan::VulkanModelPool::CreateModel()
{
	unsigned int new_index = 0;
	if (m_free_indexs.size() > 0)
	{
		// Get a index from the free array
		new_index = m_free_indexs[0];
		// Remove the index from the free array
		m_free_indexs.erase(m_free_indexs.begin());
	}
	else
	{
		new_index = m_current_index;
		m_current_index++;
		m_largest_index = m_current_index;
	}



	VulkanModel* model = new VulkanModel(this, new_index, GetVertexOffset(), GetIndexOffset(), GetIndexSize());
	m_models[new_index] = model;

	for (auto buffer = m_buffers.begin(); buffer != m_buffers.end(); buffer++)
	{
		unsigned int bufferPoolOffset = buffer->second->Allocate();
		void* data = buffer->second->GetRaw(bufferPoolOffset);
		model->SetDataPointer(buffer->first, data);

		m_model_buffer_mapping[new_index][buffer->first] = bufferPoolOffset;
	}
	m_change = true;

	Render(new_index, true);

	SetModelOffsets(new_index, GetVertexOffset(), GetIndexOffset(), GetIndexSize());

	return model;
}

Renderer::Vulkan::VulkanModel * Renderer::Vulkan::VulkanModelPool::CreateModel(unsigned int vertexOffset, unsigned int indexOffset, unsigned int indexSize)
{
	unsigned int new_index = 0;
	if (m_free_indexs.size() > 0)
	{
		// Get a index from the free array
		new_index = m_free_indexs[0];
		// Remove the index from the free array
		m_free_indexs.erase(m_free_indexs.begin());
	}
	else
	{
		new_index = m_current_index;
		m_current_index++;
		m_largest_index = m_current_index;
	}



	VulkanModel* model = new VulkanModel(this, new_index, vertexOffset, indexOffset, indexSize);
	m_models[new_index] = model;

	for (auto buffer = m_buffers.begin(); buffer != m_buffers.end(); buffer++)
	{
		unsigned int bufferPoolOffset = buffer->second->Allocate();
		void* data = buffer->second->GetRaw(bufferPoolOffset);
		model->SetDataPointer(buffer->first, data);

		m_model_buffer_mapping[new_index][buffer->first] = bufferPoolOffset;
	}
	m_change = true;

	Render(new_index, true);

	SetModelOffsets(new_index,vertexOffset, indexOffset, indexSize);

	return model;
}

Renderer::Vulkan::VulkanModel* Renderer::Vulkan::VulkanModelPool::GetModel(int index)
{
	return m_models[index];
}

void Renderer::Vulkan::VulkanModelPool::RemoveModel(VulkanModel * model)
{
	unsigned int index = model->GetModelPoolIndex();
	auto it = m_models.find(index);
	// Do we have that index and dose the model match out records
	if (it != m_models.end() && it->second == model)
	{

		model->ShouldRender(false);



		for (auto buffer = m_buffers.begin(); buffer != m_buffers.end(); buffer++)
		{
			buffer->second->UnAllocate(m_model_buffer_mapping[index][buffer->first]);
		}

		// If this model pool had any mapped buffers, remove there mapping
		auto buffer_it = m_model_buffer_mapping.find(index);
		if (buffer_it != m_model_buffer_mapping.end())
		{
			m_model_buffer_mapping.erase(buffer_it);
		}



		// Remove local model record
		m_models.erase(it);

		// Add the index to the free array
		m_free_indexs.push_back(index);

		delete model;
		model = nullptr;

		m_change = true;
	}

}

void Renderer::Vulkan::VulkanModelPool::Update()
{
	if (m_vertex_index_change)
	{
		m_vertex_index_change = false;



		if (Indexed())
		{


			if (m_model_pool_usage == ModelPoolUsage::SingleMesh)
			{
				for (int i = 0; i < m_indexed_indirect_command.size(); i++)
				{
					VkDrawIndexedIndirectCommand& indexed_indirect_command = m_indexed_indirect_command[i];
					indexed_indirect_command.indexCount = GetIndexSize();
					indexed_indirect_command.vertexOffset = GetVertexOffset();
					// Since we are in global view, we do not need to offset the offset as it is done in the command buffer creation
					//indexed_indirect_command.firstIndex = 0;
				}
			}
			else if (m_model_pool_usage == ModelPoolUsage::MultiMesh)
			{
				for (auto& it : m_models)
				{
					VulkanModel* model = it.second;
					VkDrawIndexedIndirectCommand& indexed_indirect_command = m_indexed_indirect_command[it.first];
					indexed_indirect_command.indexCount = model->GetIndexSize();
					indexed_indirect_command.vertexOffset = model->GetVertexOffset();
					indexed_indirect_command.firstIndex = model->GetIndexOffset();
				}
			}

		}
		else
		{
			for (int i = 0; i < m_vertex_indirect_command.size(); i++)
			{
				VkDrawIndirectCommand& vertex_indirect_command = m_vertex_indirect_command[i];
				vertex_indirect_command.vertexCount = GetVertexSize();
			}
		}
		m_indirect_draw_buffer->SetData(BufferSlot::Primary);
	}
	for (auto it = m_buffers.begin(); it != m_buffers.end(); it++)
	{
		it->second->GetBuffer()->SetData(BufferSlot::Primary);
	}
}


void Renderer::Vulkan::VulkanModelPool::AttachBufferPool(unsigned int index, VulkanBufferPool * buffer)
{
	m_buffers[index] = buffer;
}

void Renderer::Vulkan::VulkanModelPool::UpdateModelBuffer(unsigned int index)
{
	for (auto& it : m_models)
	{
		if (it.second != nullptr)
		{
			it.second->SetDataPointer(index, m_buffers[index]->GetRaw(m_model_buffer_mapping[it.first][index]));
		}
	}
}

void Renderer::Vulkan::VulkanModelPool::AttachDescriptorSet(unsigned int index, VulkanDescriptorSet * descriptor_set)
{
	m_descriptor_sets[index] = descriptor_set;
}

std::vector<Renderer::Vulkan::VulkanDescriptorSet*> Renderer::Vulkan::VulkanModelPool::GetDescriptorSets()
{
	std::vector<Renderer::Vulkan::VulkanDescriptorSet*> sets;
	for (auto& set : m_descriptor_sets)
	{
		sets.push_back(set.second);
	 }
	return sets;
}

unsigned int Renderer::Vulkan::VulkanModelPool::GetLargestIndex()
{
	return m_largest_index;
}

void Renderer::Vulkan::VulkanModelPool::AllowCustomScissors(bool allow)
{
	m_allow_custom_scissors = allow;
}


void Renderer::Vulkan::VulkanModelPool::SetVertexBuffer(VulkanVertexBuffer * vertex_buffer)
{
	m_vertex_buffer = vertex_buffer;
}

Renderer::Vulkan::VulkanVertexBuffer * Renderer::Vulkan::VulkanModelPool::GetVertexBuffer()
{
	return m_vertex_buffer;
}

Renderer::Vulkan::VulkanIndexBuffer * Renderer::Vulkan::VulkanModelPool::GetIndexBuffer()
{
	return m_index_buffer;
}

Renderer::Vulkan::VulkanBuffer * Renderer::Vulkan::VulkanModelPool::GetIndirectDrawBuffer()
{
	return m_indirect_draw_buffer;
}

unsigned int Renderer::Vulkan::VulkanModelPool::GetVertexOffset()
{
	return m_vertex_offset;
}

unsigned int Renderer::Vulkan::VulkanModelPool::GetIndexOffset()
{
	return m_index_offset;
}

unsigned int Renderer::Vulkan::VulkanModelPool::GetVertexSize()
{
	return m_vertex_size;
}

unsigned int Renderer::Vulkan::VulkanModelPool::GetIndexSize()
{
	return m_index_size;
}


void Renderer::Vulkan::VulkanModelPool::SetVertexOffset(unsigned int offset)
{
	m_vertex_offset = offset;
	m_vertex_index_change = true;
}

void Renderer::Vulkan::VulkanModelPool::SetIndexOffset(unsigned int offset)
{
	m_index_offset = offset;
	m_vertex_index_change = true;
}

void Renderer::Vulkan::VulkanModelPool::SetVertexSize(unsigned int size)
{
	m_vertex_size = size;
	m_vertex_index_change = true;
}

void Renderer::Vulkan::VulkanModelPool::SetIndexSize(unsigned int size)
{
	m_index_size = size;
	m_vertex_index_change = true;
}

unsigned int Renderer::Vulkan::VulkanModelPool::GetModelBufferOffset(VulkanModel * model, unsigned int buffer)
{
	return m_model_buffer_mapping[model->GetModelPoolIndex()][buffer];
}

std::map<unsigned int, Renderer::Vulkan::VulkanModel*>& Renderer::Vulkan::VulkanModelPool::GetModels()
{
	return m_models;
}

std::map<unsigned int, Renderer::Vulkan::VulkanBufferPool*>& Renderer::Vulkan::VulkanModelPool::GetBufferPools()
{
	return m_buffers;
}

void Renderer::Vulkan::VulkanModelPool::AttachToCommandBuffer(VkCommandBuffer & command_buffer, VulkanGraphicsPipeline* pipeline)
{
	for (auto it = m_descriptor_sets.begin(); it != m_descriptor_sets.end(); it++)
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

	VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(
		command_buffer,
		0,
		1,
		&m_vertex_buffer->GetBufferData(BufferSlot::Primary)->buffer,
		offsets
	);

	if (Indexed())
	{
		vkCmdBindIndexBuffer(
			command_buffer,
			m_index_buffer->GetBufferData(BufferSlot::Primary)->buffer,
			GetIndexOffset() * m_index_buffer->GetIndexSize(BufferSlot::Primary),
			VK_INDEX_TYPE_UINT32
		);
	}

	

	Renderer::NativeWindowHandle* window_handle = pipeline->GetRenderPass()->GetSwapchain()->GetNativeWindowHandle();
	
	if (!m_allow_custom_scissors)
	{
		VkRect2D scissor = VulkanInitializers::Scissor(window_handle->width, window_handle->height);
		vkCmdSetScissor(command_buffer, 0, 1, &scissor);
	}

	if (Indexed())
	{
		for (unsigned int j = 0; j < m_current_index; j++)
		{

			std::vector<VkDeviceSize> buffer_offsets;
			if (m_buffers.size() > 0)
			{
				std::vector<VkBuffer> vertex_buffers;
				for (auto buffer = m_buffers.begin(); buffer != m_buffers.end(); buffer++)
				{
					buffer_offsets.push_back((m_model_buffer_mapping[j][buffer->first] + 0) * buffer->second->GetBuffer()->GetIndexSize(BufferSlot::Primary));
					vertex_buffers.push_back(buffer->second->GetBuffer()->GetBufferData(BufferSlot::Primary)->buffer);
				}
				vkCmdBindVertexBuffers(
					command_buffer,
					1,
					(uint32_t)vertex_buffers.size(),
					vertex_buffers.data(),
					buffer_offsets.data()
				);
			}
			// Should we use the default provided scissor or the custom defined one
			if (m_allow_custom_scissors)
			{
				if (m_models[j]->UsingCustomScissor())
				{
					VkRect2D& scissor = m_models[j]->GetScissor();
					vkCmdSetScissor(command_buffer, 0, 1, &scissor);
				}
				else
				{
					VkRect2D scissor = VulkanInitializers::Scissor(window_handle->width, window_handle->height);
					vkCmdSetScissor(command_buffer, 0, 1, &scissor);
				}
			}


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
		for (unsigned int j = 0; j < m_current_index; j++)
		{

			for (auto it = m_descriptor_sets.begin(); it != m_descriptor_sets.end(); it++)
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

			// Should we use the default provided scissor or the custom defined one

			if (m_models[j]->UsingCustomScissor())
			{
				VkRect2D scissor = m_models[j]->GetScissor();
				vkCmdSetScissor(command_buffer, 0, 1, &scissor);
			}
			else
			{
				VkRect2D scissor = VulkanInitializers::Scissor(window_handle->width, window_handle->height);
				vkCmdSetScissor(command_buffer, 0, 1, &scissor);
			}
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

bool Renderer::Vulkan::VulkanModelPool::HasChanged()
{
	if (m_change)
	{
		m_change = false;
		return true;
	}
	return false;
}

void Renderer::Vulkan::VulkanModelPool::SetModelOffsets(unsigned int index, unsigned int vertexOffset, unsigned int indexOffset, unsigned int indexSize)
{
	if (index + 1 >= m_indirect_draw_buffer->GetElementCount(BufferSlot::Primary))
	{
		ResizeIndirectArray(index + m_indirect_array_padding);
	}
	auto& it = m_models.find(index);
	if (it == m_models.end())return;
	if (Indexed())
	{
		VkDrawIndexedIndirectCommand& indexed_indirect_command = m_indexed_indirect_command[index];
		indexed_indirect_command.indexCount = it->second->GetIndexSize();
		indexed_indirect_command.firstIndex = it->second->GetIndexOffset();
		indexed_indirect_command.vertexOffset = it->second->GetVertexOffset() + GetVertexOffset();
	}
	else
	{
		VkDrawIndirectCommand& vertex_indirect_command = m_vertex_indirect_command[index];
		vertex_indirect_command.firstVertex = it->second->GetVertexOffset() + GetVertexOffset();
	}

	m_indirect_draw_buffer->SetData(BufferSlot::Primary, index, 1);
}

void Renderer::Vulkan::VulkanModelPool::ResizeIndirectArray(unsigned int size)
{

	unsigned int old_size;
	unsigned int instance_size;
	if (Indexed())
	{
		instance_size = sizeof(VkDrawIndexedIndirectCommand);
		old_size = (uint32_t)m_indexed_indirect_command.size();
		m_indexed_indirect_command.resize(size);
	}
	else
	{
		instance_size = sizeof(VkDrawIndirectCommand);
		old_size = (uint32_t)m_vertex_indirect_command.size();
		m_vertex_indirect_command.resize(size);
	}
	// If the buffer is not created, create it
	if (m_indirect_draw_buffer == nullptr)
	{
		if (Indexed())
		{
			if (m_model_pool_usage == ModelPoolUsage::SingleMesh)
			{
				// Set the data for the model pool
				for (unsigned int i = 0; i < size; i++)
				{
					VkDrawIndexedIndirectCommand& indexed_indirect_command = m_indexed_indirect_command[i];
					indexed_indirect_command.indexCount = GetIndexSize();
					indexed_indirect_command.instanceCount = 0;
					indexed_indirect_command.firstIndex = 0;
					indexed_indirect_command.vertexOffset = GetVertexOffset();
					indexed_indirect_command.firstInstance = 0;
				}
			}
			else if (m_model_pool_usage == ModelPoolUsage::MultiMesh)
			{
				for (auto& it : m_models)
				{
					unsigned int index = it.first;
					// Set the data for the model
					VkDrawIndexedIndirectCommand& indexed_indirect_command = m_indexed_indirect_command[index];
					indexed_indirect_command.indexCount = it.second->GetIndexSize();
					indexed_indirect_command.instanceCount = 0;
					indexed_indirect_command.firstIndex = it.second->GetIndexOffset();
					indexed_indirect_command.vertexOffset = it.second->GetVertexOffset() + GetVertexOffset();
					indexed_indirect_command.firstInstance = 0;
				}
			}


			// Create the vulkan buffer
			m_indirect_draw_buffer = new VulkanBuffer(m_device, BufferChain::Single, m_indexed_indirect_command.data(), instance_size, size,
				VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

			m_indirect_draw_buffer->SetData(BufferSlot::Primary);
		}
		else
		{
			if (m_model_pool_usage == ModelPoolUsage::SingleMesh)
			{
				// Set the data for the model pool
				for (unsigned int i = 0; i < size; i++)
				{
					VkDrawIndirectCommand& vertex_indirect_command = m_vertex_indirect_command[i];
					vertex_indirect_command.firstInstance = i;
					vertex_indirect_command.firstVertex = GetVertexOffset();
					vertex_indirect_command.instanceCount = 0;
					vertex_indirect_command.vertexCount = GetVertexBuffer()->GetElementCount(BufferSlot::Primary);
				}
			}
			else if (m_model_pool_usage == ModelPoolUsage::MultiMesh)
			{
				// Set the data for the model pool
				for (unsigned int i = 0; i < size; i++)
				{
					VkDrawIndirectCommand& vertex_indirect_command = m_vertex_indirect_command[i];
					vertex_indirect_command.firstInstance = i;
					vertex_indirect_command.firstVertex = GetVertexOffset();
					vertex_indirect_command.instanceCount = 0;
					vertex_indirect_command.vertexCount = GetVertexBuffer()->GetElementCount(BufferSlot::Primary);
				}
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
			if (m_model_pool_usage == ModelPoolUsage::SingleMesh)
			{
				// Set the data for the model pool
				for (unsigned int i = old_size; i < size; i++)
				{
					VkDrawIndexedIndirectCommand& indexed_indirect_command = m_indexed_indirect_command[i];
					indexed_indirect_command.indexCount = GetIndexSize();
					indexed_indirect_command.instanceCount = 0;
					indexed_indirect_command.firstIndex = 0;
					indexed_indirect_command.vertexOffset = GetVertexOffset();
					indexed_indirect_command.firstInstance = i;
				}
			}
			m_indirect_draw_buffer->Resize(BufferSlot::Primary, m_indexed_indirect_command.data(), size);
		}
		else
		{
			if (m_model_pool_usage == ModelPoolUsage::SingleMesh)
			{
				for (unsigned int i = old_size; i < size; i++)
				{
					VkDrawIndirectCommand& vertex_indirect_command = m_vertex_indirect_command[i];
					vertex_indirect_command.firstInstance = i;
					vertex_indirect_command.firstVertex = GetVertexOffset();
					vertex_indirect_command.instanceCount = 0;
					vertex_indirect_command.vertexCount = GetVertexBuffer()->GetElementCount(BufferSlot::Primary);
				}
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


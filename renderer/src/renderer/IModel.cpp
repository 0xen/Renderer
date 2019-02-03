#include <renderer/IModel.hpp>
#include <renderer\IModelPool.hpp>

Renderer::IModel::IModel(unsigned int model_pool_index)
{
	m_model_pool_index = model_pool_index;
}

void Renderer::IModel::SetDataPointer(unsigned int index, void * data)
{
	m_data_pointers[index] = data;
}

void Renderer::IModel::SetData(unsigned int index, void * data, unsigned int size)
{
	memcpy(m_data_pointers[index], data, size);
}

unsigned int Renderer::IModel::GetModelPoolIndex()
{
	return m_model_pool_index;
}

void Renderer::IModel::Remove()
{
	GetModelPool()->RemoveModel(this);
}

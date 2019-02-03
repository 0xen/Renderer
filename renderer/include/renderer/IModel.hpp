#pragma once

#include <map>

namespace Renderer
{
	class IModelPool;
	class IModel
	{
	public:
		IModel(unsigned int model_pool_index);
		void SetDataPointer(unsigned int index, void* data);
		void SetData(unsigned int index, void* data, unsigned int size);
		template <class T>
		void SetData(unsigned int index, T data);
		template <class T>
		T& GetData(unsigned int index);
		unsigned int GetModelPoolIndex();
		virtual void ShouldRender(bool render) = 0;
		virtual bool Rendering() = 0;
		virtual IModelPool* GetModelPool() = 0;
		void Remove();
	protected:
		unsigned int m_model_pool_index;
		std::map<unsigned int, void*> m_data_pointers;
	};
	template<class T>
	inline void IModel::SetData(unsigned int index, T data)
	{
		memcpy(m_data_pointers[index], &data, sizeof(T));
	}
	template<class T>
	inline T& IModel::GetData(unsigned int index)
	{
		return *static_cast<T*>(m_data_pointers[index]);
	}
}
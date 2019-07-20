#pragma once

#include <map>

namespace Renderer
{

	namespace Vulkan
	{
		class VulkanModelPool;
		class VulkanModel
		{
		public:
			VulkanModel(VulkanModelPool* pool, unsigned int model_pool_index);


			void SetDataPointer(unsigned int index, void* data);
			void SetData(unsigned int index, void* data, unsigned int size);
			template <class T>
			void SetData(unsigned int index, T data);
			template <class T>
			T& GetData(unsigned int index);
			unsigned int GetModelPoolIndex();
			void Remove();


			virtual void ShouldRender(bool render);
			virtual bool Rendering();
			virtual VulkanModelPool* GetModelPool();
		private:
			unsigned int m_model_pool_index;
			std::map<unsigned int, void*> m_data_pointers;
			VulkanModelPool * m_pool;
			bool m_rendering;
		};
		template<class T>
		inline void VulkanModel::SetData(unsigned int index, T data)
		{
			memcpy(m_data_pointers[index], &data, sizeof(T));
		}
		template<class T>
		inline T& VulkanModel::GetData(unsigned int index)
		{
			return *static_cast<T*>(m_data_pointers[index]);
		}
	}
}
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
			VulkanModel(VulkanModelPool* pool, unsigned int model_pool_index, unsigned int vertexOffset, unsigned int indexOffset, unsigned int indexSize);


			void SetDataPointer(unsigned int index, void* data);
			void SetData(unsigned int index, void* data, unsigned int size);
			template <class T>
			void SetData(unsigned int index, T data);
			template <class T>
			T& GetData(unsigned int index);
			unsigned int GetModelPoolIndex();
			void Remove();
			void ShouldRender(bool render);
			bool Rendering();
			VulkanModelPool* GetModelPool();
			unsigned int GetVertexOffset();
			unsigned int GetIndexOffset();
			unsigned int GetIndexSize();
			void SetVertexOffset(unsigned int offset);
			void SetIndexOffset(unsigned int offset);
			void SetIndexSize(unsigned int size);
		private:
			unsigned int m_model_pool_index;
			unsigned int m_vertexOffset;
			unsigned int m_indexOffset;
			unsigned int m_indexSize;
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
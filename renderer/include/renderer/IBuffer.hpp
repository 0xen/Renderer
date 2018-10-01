#pragma once

namespace Renderer
{
	class IBuffer
	{
	public:
		virtual void SetData() = 0;
		virtual void SetData(unsigned int count) = 0;
		virtual void SetData(unsigned int startIndex, unsigned int count) = 0;

		virtual void Resize(void * dataPtr, unsigned int elementCount) = 0;
		
		unsigned int GetIndexSize();
		unsigned int GetElementCount();
		void* GetDataPointer();
	protected:
		void* m_dataPtr;
		unsigned int m_bufferSize;
		unsigned int m_indexSize;
		unsigned int m_elementCount;
	};
}
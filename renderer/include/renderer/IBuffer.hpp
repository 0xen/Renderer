#pragma once

namespace Renderer
{
	enum BufferChain
	{
		Single = 0,
		Double = 1,
		Tripple = 2,
		BUFFER_CHAIN_MAX = 3
	};
	enum BufferSlot
	{
		Primary = 0,
		Secondery = 1,
		Tertiary = 2,
		BUFFER_SLOT_MAX = 3
	};

	class IBuffer
	{
	public:
		IBuffer(BufferChain level);
		virtual ~IBuffer();
		virtual void SetData(BufferSlot slot) = 0;
		virtual void SetData(BufferSlot slot, unsigned int count) = 0;
		virtual void SetData(BufferSlot slot, unsigned int startIndex, unsigned int count) = 0;

		virtual void Resize(BufferSlot slot, void * dataPtr, unsigned int elementCount) = 0;

		virtual void Swap(BufferSlot s1, BufferSlot s2);
		
		unsigned int GetIndexSize(BufferSlot slot);
		unsigned int GetElementCount(BufferSlot slot);
		void* GetDataPointer(BufferSlot slot);
	protected:
		struct BufferLocalAllocation
		{
			void* dataPtr = nullptr;
			unsigned int bufferSize;
			unsigned int indexSize;
			unsigned int elementCount;
		};
		BufferLocalAllocation* m_local_allocation = nullptr;
		BufferChain m_level;
	};
}
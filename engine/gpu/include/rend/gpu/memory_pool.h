#pragma once

#include "rend/core/result.h"
#include "rend/gpu/buffer.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace rend::gpu {

class Device;

// A suballocation out of a memory pool. Plain offsets — draw commands
// carry them as firstIndex/vertexOffset, so no per-object binds.
struct BufferSlice {
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
};

// The geometry memory pool (see ARCHITECTURE.md): all vertex/index data
// suballocated from one device-local buffer that is bound once. First-fit
// free list with coalescing on free; defrag is a later, separate concern
// (not yet built). Fixed capacity — growing would invalidate every offset the
// indirect entries reference.
class MemoryPool {
public:
    // extraUsage: additional VkBufferUsageFlags (e.g. device address + AS
    // build input when the geometry also feeds ray tracing).
    static Result<std::unique_ptr<MemoryPool>> create(const Device& device, std::uint64_t capacity,
                                                      std::uint32_t extraUsage = 0);

    Result<BufferSlice> allocate(std::uint64_t size, std::uint64_t alignment = 16);
    void free(const BufferSlice& slice);

    const Buffer& buffer() const { return *buffer_; }
    std::uint64_t capacity() const { return buffer_->size(); }
    std::uint64_t usedBytes() const { return usedBytes_; }
    std::uint32_t allocationCount() const { return allocationCount_; }

private:
    MemoryPool() = default;

    struct FreeBlock {
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
    };

    std::unique_ptr<Buffer> buffer_;
    std::vector<FreeBlock> freeList_; // sorted by offset, no adjacent blocks
    std::uint64_t usedBytes_ = 0;
    std::uint32_t allocationCount_ = 0;
};

} // namespace rend::gpu

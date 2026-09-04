#pragma once

#include "rend/core/result.h"
#include "rend/gpu/buffer.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace rend::gpu {

class Device;

// A suballocation out of the mega-buffer. Plain offsets — draw commands
// carry them as firstIndex/vertexOffset, so no per-object binds.
struct BufferSlice {
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
};

// The geometry mega-buffer (see ARCHITECTURE.md): all vertex/index data
// suballocated from one device-local buffer that is bound once. First-fit
// free list with coalescing on free; defrag is a later, separate concern
// (roadmap #9). Fixed capacity — growing would invalidate every offset the
// indirect entries reference.
class MegaBuffer {
public:
    static Result<std::unique_ptr<MegaBuffer>> create(const Device& device, std::uint64_t capacity);

    Result<BufferSlice> allocate(std::uint64_t size, std::uint64_t alignment = 16);
    void free(const BufferSlice& slice);

    const Buffer& buffer() const { return *buffer_; }
    std::uint64_t capacity() const { return buffer_->size(); }
    std::uint64_t usedBytes() const { return usedBytes_; }
    std::uint32_t allocationCount() const { return allocationCount_; }

private:
    MegaBuffer() = default;

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

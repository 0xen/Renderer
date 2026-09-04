#include "rend/gpu/memory_pool.h"

#include "rend/core/log.h"
#include "rend/gpu/device.h"

#include <volk.h>

#include <algorithm>
#include <format>

namespace rend::gpu {

Result<std::unique_ptr<MemoryPool>> MemoryPool::create(const Device& device,
                                                       std::uint64_t capacity) {
    auto bufferResult =
        Buffer::create(device, {
                                   .size = capacity,
                                   // Vertex + index + storage (object data reads), transfer both
                                   // ways (uploads now, defrag moves later).
                                   .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                   .location = MemoryLocation::DeviceLocal,
                                   .sharedWithTransferQueue = true,
                               });
    if (!bufferResult) {
        return bufferResult.error();
    }

    auto pool = std::unique_ptr<MemoryPool>(new MemoryPool());
    pool->buffer_ = std::move(bufferResult).value();
    pool->freeList_.push_back({.offset = 0, .size = capacity});
    log::info("Geometry memory pool created: {} MiB device-local", capacity / (1024 * 1024));
    return pool;
}

Result<BufferSlice> MemoryPool::allocate(std::uint64_t size, std::uint64_t alignment) {
    if (size == 0) {
        return Error{"Memory pool allocation size must be non-zero"};
    }
    for (std::size_t i = 0; i < freeList_.size(); ++i) {
        FreeBlock& block = freeList_[i];
        const std::uint64_t aligned = (block.offset + alignment - 1) & ~(alignment - 1);
        const std::uint64_t padding = aligned - block.offset;
        if (padding + size > block.size) {
            continue;
        }
        // First fit. The padding stays in the free list as its own block so
        // it can coalesce back when the neighbour is freed.
        BufferSlice slice{.offset = aligned, .size = size};
        const std::uint64_t remaining = block.size - padding - size;
        if (padding > 0 && remaining > 0) {
            block.size = padding;
            freeList_.insert(freeList_.begin() + static_cast<std::ptrdiff_t>(i) + 1,
                             {.offset = aligned + size, .size = remaining});
        } else if (padding > 0) {
            block.size = padding;
        } else if (remaining > 0) {
            block.offset = aligned + size;
            block.size = remaining;
        } else {
            freeList_.erase(freeList_.begin() + static_cast<std::ptrdiff_t>(i));
        }
        usedBytes_ += size;
        ++allocationCount_;
        return slice;
    }
    return Error{std::format(
        "Geometry memory pool exhausted: {} bytes requested, {} of {} in use ({} free blocks)",
        size, usedBytes_, capacity(), freeList_.size())};
}

void MemoryPool::free(const BufferSlice& slice) {
    if (slice.size == 0) {
        return;
    }
    const auto next = std::ranges::lower_bound(freeList_, slice.offset, {}, &FreeBlock::offset);
    const auto index = static_cast<std::size_t>(next - freeList_.begin());

    // Coalesce with the neighbours where they touch.
    const bool mergesPrev =
        index > 0 && freeList_[index - 1].offset + freeList_[index - 1].size == slice.offset;
    const bool mergesNext =
        index < freeList_.size() && slice.offset + slice.size == freeList_[index].offset;
    if (mergesPrev && mergesNext) {
        freeList_[index - 1].size += slice.size + freeList_[index].size;
        freeList_.erase(freeList_.begin() + static_cast<std::ptrdiff_t>(index));
    } else if (mergesPrev) {
        freeList_[index - 1].size += slice.size;
    } else if (mergesNext) {
        freeList_[index].offset = slice.offset;
        freeList_[index].size += slice.size;
    } else {
        freeList_.insert(freeList_.begin() + static_cast<std::ptrdiff_t>(index),
                         {.offset = slice.offset, .size = slice.size});
    }
    usedBytes_ -= slice.size;
    --allocationCount_;
}

} // namespace rend::gpu

#pragma once

#include "rend/core/result.h"

#include <cstdint>
#include <memory>
#include <vector>

typedef struct VkCommandPool_T* VkCommandPool;
typedef struct VkCommandBuffer_T* VkCommandBuffer;
typedef struct VkFence_T* VkFence;

namespace rend::gpu {

class Buffer;
class Device;

// Batched CPU→VRAM uploads on the dedicated transfer queue (graphics queue
// fallback). stage() copies bytes into a growing host-visible staging
// buffer immediately; flush() submits every pending copy in one command
// buffer and blocks until the GPU is done. Synchronous by design — async
// streaming with semaphore chaining lands with roadmap #9; destination
// buffers currently opt into concurrent sharing instead of ownership
// transfers (BufferDesc::sharedWithTransferQueue).
class TransferContext {
public:
    static Result<std::unique_ptr<TransferContext>> create(const Device& device);
    ~TransferContext();

    TransferContext(const TransferContext&) = delete;
    TransferContext& operator=(const TransferContext&) = delete;

    Result<void> stage(const Buffer& dst, std::uint64_t dstOffset, const void* data,
                       std::uint64_t size);
    Result<void> flush();

    std::uint64_t pendingBytes() const { return stagingUsed_; }

private:
    TransferContext() = default;
    Result<void> ensureStagingCapacity(std::uint64_t required);

    struct PendingCopy {
        const Buffer* dst = nullptr;
        std::uint64_t srcOffset = 0;
        std::uint64_t dstOffset = 0;
        std::uint64_t size = 0;
    };

    const Device* device_ = nullptr;
    VkCommandPool pool_ = nullptr;
    VkCommandBuffer cmd_ = nullptr;
    VkFence fence_ = nullptr;
    std::unique_ptr<Buffer> staging_;
    std::uint64_t stagingUsed_ = 0;
    std::vector<PendingCopy> pending_;
};

} // namespace rend::gpu

#pragma once

#include "rend/core/result.h"

#include <cstdint>
#include <memory>

namespace rend::gpu {

class Buffer;
class Device;

// Batched CPU→VRAM uploads on the dedicated transfer queue (graphics queue
// fallback). stage() copies bytes into a growing host-visible staging
// buffer immediately; flush() submits every pending copy in one command
// buffer and blocks until the GPU is done. Synchronous by design — async
// streaming with semaphore chaining is a later step; destination
// buffers currently opt into concurrent sharing instead of ownership
// transfers (BufferDesc::sharedWithTransferQueue).
class TransferContext {
public:
    static Result<std::unique_ptr<TransferContext>> create(const Device& device);
    virtual ~TransferContext() = default;

    TransferContext(const TransferContext&) = delete;
    TransferContext& operator=(const TransferContext&) = delete;

    virtual Result<void> stage(const Buffer& dst, std::uint64_t dstOffset, const void* data,
                               std::uint64_t size) = 0;
    virtual Result<void> flush() = 0;

    virtual std::uint64_t pendingBytes() const = 0;

protected:
    TransferContext() = default;
};

} // namespace rend::gpu

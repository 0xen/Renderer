#pragma once

#include "rend/core/result.h"
#include "rend/gpu/memory_tracker.h"

#include <cstdint>
#include <memory>

namespace rend::gpu {

class Device;

// Buffer usage bits (backend-neutral vocabulary; each backend maps them).
inline constexpr std::uint32_t kUsageTransferDst = 0x2;
inline constexpr std::uint32_t kUsageStorage = 0x20;
inline constexpr std::uint32_t kUsageIndirect = 0x100;
inline constexpr std::uint32_t kUsageShaderDeviceAddress = 0x20000;
inline constexpr std::uint32_t kUsageAccelBuildInput = 0x80000; // AS build read-only input

enum class MemoryLocation {
    DeviceLocal, // VRAM; filled via transfer
    HostVisible, // CPU-writable (host-coherent), persistently mapped
};

struct BufferDesc {
    std::uint64_t size = 0;
    std::uint32_t usage = 0; // kUsage* bits
    MemoryLocation location = MemoryLocation::DeviceLocal;
    // Share with the dedicated transfer family (concurrent sharing mode) so
    // uploads need no queue-ownership transfers. Revisited with async
    // streaming/defrag (roadmap #9), where explicit transfers pay off.
    bool sharedWithTransferQueue = false;
};

// One GPU buffer plus its dedicated allocation, nothing more (see
// ARCHITECTURE.md base object classes). Suballocation, residency and
// meaning all live above; this class stays policy-free — which is fine
// allocation-count-wise because composition happens in big buffers.
class Buffer {
public:
    static Result<std::unique_ptr<Buffer>> create(const Device& device, const BufferDesc& desc);
    virtual ~Buffer() = default;

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    std::uint64_t size() const { return size_; }
    // Persistent mapping; non-null only for HostVisible buffers.
    void* mapped() const { return mapped_; }

protected:
    Buffer() = default;

    std::uint64_t size_ = 0;
    void* mapped_ = nullptr;
};

} // namespace rend::gpu

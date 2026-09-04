#pragma once

#include "rend/core/result.h"

#include <cstdint>
#include <memory>

typedef struct VkBuffer_T* VkBuffer;
typedef struct VkDeviceMemory_T* VkDeviceMemory;

namespace rend::gpu {

class Device;

// VkBufferUsageFlags bits callers need without including Vulkan headers.
inline constexpr std::uint32_t kUsageTransferDst = 0x2;
inline constexpr std::uint32_t kUsageStorage = 0x20;
inline constexpr std::uint32_t kUsageIndirect = 0x100;

enum class MemoryLocation {
    DeviceLocal, // VRAM; filled via transfer
    HostVisible, // CPU-writable (host-coherent), persistently mapped
};

struct BufferDesc {
    std::uint64_t size = 0;
    std::uint32_t usage = 0; // VkBufferUsageFlags
    MemoryLocation location = MemoryLocation::DeviceLocal;
    // Share with the dedicated transfer family (concurrent sharing mode) so
    // uploads need no queue-ownership transfers. Revisited with async
    // streaming/defrag (roadmap #9), where explicit transfers pay off.
    bool sharedWithTransferQueue = false;
};

// One VkBuffer plus its dedicated allocation, nothing more (see
// ARCHITECTURE.md base object classes). Suballocation, residency and
// meaning all live above; this class stays policy-free — which is fine
// allocation-count-wise because composition happens in big buffers.
class Buffer {
public:
    static Result<std::unique_ptr<Buffer>> create(const Device& device, const BufferDesc& desc);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    VkBuffer handle() const { return buffer_; }
    std::uint64_t size() const { return size_; }
    // Persistent mapping; non-null only for HostVisible buffers.
    void* mapped() const { return mapped_; }

private:
    Buffer() = default;

    const Device* device_ = nullptr;
    VkBuffer buffer_ = nullptr;
    VkDeviceMemory memory_ = nullptr;
    std::uint64_t size_ = 0;
    void* mapped_ = nullptr;
};

} // namespace rend::gpu

#pragma once

#include "rend/core/result.h"

#include <cstdint>
#include <memory>
#include <span>

typedef struct VkAccelerationStructureKHR_T* VkAccelerationStructureKHR;

namespace rend::gpu {

class Buffer;
class Device;

// One BLAS or TLAS plus its storage buffer — the ray-tracing sibling of
// Buffer/Image (see ARCHITECTURE.md base object classes). Builds are
// synchronous one-shots on the graphics queue: acceleration structures are
// load-time work until dynamic scenes need refits.
class AccelerationStructure {
public:
    // Triangle range inside a geometry buffer (offsets in bytes, uint32
    // indices). Positions must be the first attribute at each stride step.
    struct TriangleGeometry {
        const Buffer* buffer = nullptr;
        std::uint64_t vertexOffset = 0;
        std::uint32_t vertexStride = 0;
        std::uint32_t vertexCount = 0;
        std::uint64_t indexOffset = 0;
        std::uint32_t indexCount = 0;
        // Opaque geometries never surface as ray-query candidates; leave
        // false for alpha-masked or blended surfaces so shaders can run
        // the any-hit alpha test / transparency march on them.
        bool opaque = true;
    };

    struct Instance {
        const AccelerationStructure* blas = nullptr;
        std::uint32_t customIndex = 0; // lands in InstanceID()
    };

    // One BLAS holding every given triangle range. Requires
    // Feature::AccelerationStructure and buffers created with device
    // address + AS-build-input usage.
    static Result<std::unique_ptr<AccelerationStructure>> buildBottomLevel(
        const Device& device, std::span<const TriangleGeometry> geometries);

    // A TLAS over the given instances (identity transforms: our geometry
    // is world-space baked).
    static Result<std::unique_ptr<AccelerationStructure>> buildTopLevel(
        const Device& device, std::span<const Instance> instances);

    ~AccelerationStructure();

    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;

    VkAccelerationStructureKHR handle() const { return as_; }

private:
    AccelerationStructure() = default;

    const Device* device_ = nullptr;
    VkAccelerationStructureKHR as_ = nullptr;
    std::unique_ptr<Buffer> storage_;
};

} // namespace rend::gpu

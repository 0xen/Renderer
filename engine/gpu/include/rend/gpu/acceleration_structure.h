#pragma once

#include "rend/core/result.h"

#include <cstdint>
#include <memory>
#include <span>

typedef struct VkAccelerationStructureKHR_T* VkAccelerationStructureKHR;
typedef struct VkCommandBuffer_T* VkCommandBuffer;

namespace rend::gpu {

class Buffer;
class Device;

// One BLAS or TLAS plus its storage buffer — the ray-tracing sibling of
// Buffer/Image (see ARCHITECTURE.md base object classes). Builds are
// synchronous one-shots on the graphics queue; structures built with
// allowUpdate additionally keep an update scratch buffer alive so callers
// can record per-frame in-place refits (animated geometry).
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
        const Device& device, std::span<const TriangleGeometry> geometries,
        bool allowUpdate = false);

    // A TLAS over the given instances (identity transforms: our geometry
    // is world-space baked).
    static Result<std::unique_ptr<AccelerationStructure>> buildTopLevel(
        const Device& device, std::span<const Instance> instances, bool allowUpdate = false);

    // Records an in-place refit (mode UPDATE, src == dst) into cmd. The
    // BLAS overload takes fresh geometry ranges — same count and primitive
    // counts as the build, but the vertex data may point elsewhere (posed
    // per-slot regions). The TLAS overload rereads the instance buffer kept
    // from the build; call it after a BLAS refit so the instance AABBs
    // follow. Both require allowUpdate at build time; the caller owns the
    // barriers around the build stages.
    void recordRefit(VkCommandBuffer cmd, std::span<const TriangleGeometry> geometries) const;
    void recordRefit(VkCommandBuffer cmd) const;

    ~AccelerationStructure();

    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;

    VkAccelerationStructureKHR handle() const { return as_; }

private:
    AccelerationStructure() = default;

    const Device* device_ = nullptr;
    VkAccelerationStructureKHR as_ = nullptr;
    std::unique_ptr<Buffer> storage_;
    // Refit state (allowUpdate builds only): scratch sized for UPDATE mode,
    // and for a TLAS the live instance buffer the update rereads.
    std::unique_ptr<Buffer> updateScratch_;
    std::unique_ptr<Buffer> instances_;
    std::uint32_t instanceCount_ = 0;
};

} // namespace rend::gpu

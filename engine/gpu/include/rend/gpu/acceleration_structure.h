#pragma once

#include "rend/core/result.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace rend::gpu {

class Buffer;
class CommandContext;
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
        const AccelerationStructure* blas = nullptr; // null = inactive entry
        std::uint32_t customIndex = 0; // lands in InstanceID()
        // Column-major 4x4 (math::Mat4 layout); the translation-free bottom
        // row is dropped for the 3x4 Vulkan instance transform.
        std::array<float, 16> transform{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    };

    // One BLAS holding every given triangle range. Requires
    // Feature::AccelerationStructure and buffers created with device
    // address + AS-build-input usage.
    static Result<std::unique_ptr<AccelerationStructure>> buildBottomLevel(
        const Device& device, std::span<const TriangleGeometry> geometries,
        bool allowUpdate = false);

    // A TLAS over the given instances.
    static Result<std::unique_ptr<AccelerationStructure>> buildTopLevel(
        const Device& device, std::span<const Instance> instances, bool allowUpdate = false);

    // A TLAS with a FIXED instance capacity and one host-visible instance
    // region per frame slot: the CPU rewrites a slot's region each frame
    // (writeInstances — unused capacity entries are inactive null-BLAS
    // rows) and recordRebuild() re-builds the structure in place from it.
    // The recorded primitive count is always `capacity`, so instance
    // count/transform changes never touch command recordings, descriptors
    // or the AS handle. The initial synchronous build seeds every slot
    // with `initial`.
    static Result<std::unique_ptr<AccelerationStructure>> buildTopLevelDynamic(
        const Device& device, std::span<const Instance> initial, std::uint32_t capacity,
        std::uint32_t slotCount);

    // Rewrites one slot's instance region (dynamic TLAS only). Call only
    // between the slot's fence wait and its submission; entries past
    // instances.size() up to capacity become inactive.
    virtual void writeInstances(std::uint32_t slot, std::span<const Instance> instances) = 0;

    // Records an in-place full rebuild from the slot's instance region
    // (dynamic TLAS only). The caller owns the surrounding barriers.
    virtual void recordRebuild(CommandContext& cmd, std::uint32_t slot) const = 0;

    // Records an in-place refit (mode UPDATE, src == dst) into cmd. The
    // BLAS overload takes fresh geometry ranges — same count and primitive
    // counts as the build, but the vertex data may point elsewhere (posed
    // per-slot regions). The TLAS overload rereads the instance buffer kept
    // from the build; call it after a BLAS refit so the instance AABBs
    // follow. Both require allowUpdate at build time; the caller owns the
    // barriers around the build stages.
    virtual void recordRefit(CommandContext& cmd,
                             std::span<const TriangleGeometry> geometries) const = 0;
    virtual void recordRefit(CommandContext& cmd) const = 0;

    virtual ~AccelerationStructure() = default;

    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;

    // Device address of this structure (BLAS: what TLAS instances
    // reference). Cached at build so per-frame instance writes make no
    // API calls.
    std::uint64_t deviceAddress() const { return deviceAddress_; }

protected:
    AccelerationStructure() = default;

    std::uint64_t deviceAddress_ = 0;
};

} // namespace rend::gpu

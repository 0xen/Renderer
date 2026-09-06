#pragma once

#include "rend/core/result.h"

#include <cstdint>
#include <memory>

typedef struct VkAccelerationStructureKHR_T* VkAccelerationStructureKHR;
typedef struct VkDescriptorSetLayout_T* VkDescriptorSetLayout;
typedef struct VkDescriptorSet_T* VkDescriptorSet;
typedef struct VkDescriptorPool_T* VkDescriptorPool;
typedef struct VkSampler_T* VkSampler;
typedef struct VkBuffer_T* VkBuffer;
typedef struct VkImageView_T* VkImageView;

namespace rend::gpu {

class Device;

// The bindless table (see ARCHITECTURE.md): one descriptor set bound once,
// holding everything shaders index dynamically. Bindings:
//   0 — object data SSBO (vertex + fragment)
//   1 — sampled image array [maxTextures] (fragment; partially bound,
//       update-after-bind, so texture slots stream in over time)
//   2 — one trilinear repeat sampler (fragment), owned here
//   3 — draw templates SSBO (compute: the cull pass input)
//   4 — compacted draws SSBO (compute: the cull pass output)
//   5 — draw counts SSBO (compute: one uint32 per frame slot)
//   6 — camera SSBO (vertex + compute: one viewProj per frame slot)
//   7 — light SSBO (vertex + fragment: per-slot light data + matrix)
//   8 — shadow cascade maps [4] (fragment; depth images the scene samples)
//   9 — shadow comparison sampler (fragment; PCF, owned here)
//   10 — scene TLAS (fragment; only on devices with Feature::RayQuery)
//   11 — geometry pool raw bytes (fragment; RayQuery only — hit-attribute
//        fetch for traced primary rays)
//   12 — per-object geometry info SSBO (fragment; RayQuery only —
//        firstIndex/vertexOffset per BLAS geometry index)
//   13 — geometry pool raw bytes, writable (compute; the skinning pass's
//        source + destination vertices)
//   14 — skin vertex attributes SSBO (compute; packed joints + weights)
//   15 — joint matrices SSBO (compute; per frame slot)
//   16 — morph target deltas SSBO (compute; pos+normal per vertex/target)
//   17 — morph weights SSBO (compute; per frame slot)
//   18 — reflection probe cubemap (fragment; the raster reflection tier
//        samples it by direction, mip = roughness)
//   19 — per-object world transforms (vertex; per-camera-slot regions)
//   20 — instance rows SSBO (vertex; {objectIndex, transformIndex} per
//        drawn instance — SV_InstanceID resolves through it)
//   21 — frustum-culled compacted draws SSBO (compute; the scene pass's
//        stream — shadow passes keep drawing binding 4's list)
//   22 — per-object AABBs SSBO (compute; per-slot regions, min.w = 1
//        marks always-visible entries, max.w = 1 marks local bounds
//        tested per instance)
//   23 — instance rows again, writable (compute; same buffer as 20 — the
//        cull pass compacts partially visible draws' surviving rows into
//        per-slot scratch regions above the canonical rows)
class DescriptorTable {
public:
    static Result<std::unique_ptr<DescriptorTable>> create(const Device& device,
                                                           std::uint32_t maxTextures);
    ~DescriptorTable();

    DescriptorTable(const DescriptorTable&) = delete;
    DescriptorTable& operator=(const DescriptorTable&) = delete;

    VkDescriptorSetLayout layout() const { return layout_; }
    VkDescriptorSet set() const { return set_; }

    void writeObjectBuffer(VkBuffer buffer, std::uint64_t range);
    void writeTexture(std::uint32_t index, VkImageView view);
    // Storage-buffer bindings (3-7); binding picks which, see class comment.
    void writeStorageBuffer(std::uint32_t binding, VkBuffer buffer, std::uint64_t range);
    // Binding 8: one cascade's depth image the scene pass samples.
    void writeShadowMap(std::uint32_t cascade, VkImageView view);
    // Binding 18: the reflection probe's cube view.
    void writeProbe(VkImageView view);
    // Binding 10 (RayQuery devices only): the scene TLAS.
    void writeAccelerationStructure(VkAccelerationStructureKHR tlas);

private:
    DescriptorTable() = default;

    const Device* device_ = nullptr;
    VkDescriptorPool pool_ = nullptr;
    VkDescriptorSetLayout layout_ = nullptr;
    VkDescriptorSet set_ = nullptr;
    VkSampler sampler_ = nullptr;
    VkSampler shadowSampler_ = nullptr; // comparison (PCF) sampler, binding 9
};

} // namespace rend::gpu

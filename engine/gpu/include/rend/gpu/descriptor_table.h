#pragma once

#include "rend/core/result.h"

#include <cstdint>
#include <memory>

typedef struct VkDescriptorSetLayout_T* VkDescriptorSetLayout;
typedef struct VkDescriptorSet_T* VkDescriptorSet;
typedef struct VkDescriptorPool_T* VkDescriptorPool;
typedef struct VkSampler_T* VkSampler;

namespace rend::gpu {

class AccelerationStructure;
class Buffer;
class Device;
class Image;

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
//   24 — transparent draw stream SSBO (compute; the blend pass's list)
//   25 — mesh LOD tables SSBO (compute; one global region)
//   26 — occlusion visibility SSBO (fragment + compute; per-slot regions)
//   27 — point-light shadow cubes [16] (fragment; R32F distance)
//   28-31 — deferred G-buffer targets (fragment; albedo / world normal /
//        material params / view depth, rewritten on swapchain recreate)
//   32-36 — RT hit remap, refined OBBs, row->entry map, scene color, LDR
//        intermediate (descriptor_table.cpp is the authority)
//   kUserBindingBase.. — caller-declared bindings (DescriptorTableDesc):
//        the storage buffers first, then the sampled images, in the
//        order the desc counted them. userStorageBinding(i) /
//        userSampledImageBinding(i) give the binding numbers.
struct DescriptorTableDesc {
    std::uint32_t maxTextures = 0;
    // Extra bindings for the caller's own passes (frame passes, custom
    // shaders): storage buffers visible to vertex + fragment + compute,
    // sampled images visible to fragment + compute. All partially bound.
    std::uint32_t userStorageBuffers = 0;
    std::uint32_t userSampledImages = 0;
};

class DescriptorTable {
public:
    static constexpr std::uint32_t kUserBindingBase = 40;

    static Result<std::unique_ptr<DescriptorTable>> create(const Device& device,
                                                           const DescriptorTableDesc& desc);
    static Result<std::unique_ptr<DescriptorTable>> create(const Device& device,
                                                           std::uint32_t maxTextures) {
        return create(device, DescriptorTableDesc{.maxTextures = maxTextures});
    }
    ~DescriptorTable();

    // Binding numbers of the caller-declared bindings; write them with
    // writeStorageBuffer / writeSampledImage(binding, 0, view).
    std::uint32_t userStorageBinding(std::uint32_t index) const {
        return kUserBindingBase + index;
    }
    std::uint32_t userSampledImageBinding(std::uint32_t index) const {
        return kUserBindingBase + userStorageBuffers_ + index;
    }
    std::uint32_t userStorageBuffers() const { return userStorageBuffers_; }
    std::uint32_t userSampledImages() const { return userSampledImages_; }

    DescriptorTable(const DescriptorTable&) = delete;
    DescriptorTable& operator=(const DescriptorTable&) = delete;

    VkDescriptorSetLayout layout() const { return layout_; }
    VkDescriptorSet set() const { return set_; }

    // Buffer bindings take the whole buffer unless `range` (bytes) is given.
    void writeObjectBuffer(const Buffer& buffer, std::uint64_t range = 0);
    void writeTexture(std::uint32_t index, const Image& image);
    // Storage-buffer bindings; binding picks which, see class comment.
    void writeStorageBuffer(std::uint32_t binding, const Buffer& buffer, std::uint64_t range = 0);
    // Binding 8: one cascade's depth image the scene pass samples.
    void writeShadowMap(std::uint32_t cascade, const Image& image);
    // Binding 18: the reflection probe's cube image.
    void writeProbe(const Image& image);
    // Binding 27: one point light's shadow-distance cube (index = the
    // light's slot, 0..15). Not update-after-bind — idle around writes.
    void writePointShadowMap(std::uint32_t index, const Image& image);
    // Any SAMPLED_IMAGE binding/array element (SHADER_READ_ONLY layout);
    // used for the G-buffer targets 28-31. Not update-after-bind — idle
    // around writes.
    void writeSampledImage(std::uint32_t binding, std::uint32_t index, const Image& image);
    // Binding 10 (RayQuery devices only): the scene TLAS.
    void writeAccelerationStructure(const AccelerationStructure& tlas);

private:
    DescriptorTable() = default;

    const Device* device_ = nullptr;
    VkDescriptorPool pool_ = nullptr;
    VkDescriptorSetLayout layout_ = nullptr;
    VkDescriptorSet set_ = nullptr;
    VkSampler sampler_ = nullptr;
    VkSampler shadowSampler_ = nullptr; // comparison (PCF) sampler, binding 9
    std::uint32_t userStorageBuffers_ = 0;
    std::uint32_t userSampledImages_ = 0;
};

} // namespace rend::gpu

#pragma once

#include "rend/core/result.h"

#include <cstdint>
#include <memory>

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
    // Cull-pass bindings (3-5); binding picks which, see the class comment.
    void writeStorageBuffer(std::uint32_t binding, VkBuffer buffer, std::uint64_t range);

private:
    DescriptorTable() = default;

    const Device* device_ = nullptr;
    VkDescriptorPool pool_ = nullptr;
    VkDescriptorSetLayout layout_ = nullptr;
    VkDescriptorSet set_ = nullptr;
    VkSampler sampler_ = nullptr;
};

} // namespace rend::gpu

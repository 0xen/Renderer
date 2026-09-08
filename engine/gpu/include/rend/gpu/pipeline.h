#pragma once

#include "rend/core/result.h"

#include <cstdint>
#include <memory>
#include <vector>

typedef struct VkPipeline_T* VkPipeline;
typedef struct VkPipelineLayout_T* VkPipelineLayout;
typedef struct VkDescriptorSetLayout_T* VkDescriptorSetLayout;

namespace rend::gpu {

class Device;
class Shader;

// VkFormat values callers need without including Vulkan headers; the
// pipeline XML's neutral format vocabulary maps onto these.
inline constexpr std::uint32_t kFormatR32G32Sfloat = 103;
inline constexpr std::uint32_t kFormatR32G32B32Sfloat = 106;
inline constexpr std::uint32_t kFormatD32Sfloat = 126;

struct VertexAttribute {
    std::uint32_t location = 0;
    std::uint32_t format = 0; // VkFormat
    std::uint32_t offset = 0;
};

struct GraphicsPipelineDesc {
    const Shader* vertexShader = nullptr;
    const Shader* fragmentShader = nullptr;
    const char* vertexEntryPoint = "main";
    const char* fragmentEntryPoint = "main";
    // VkFormat of the single color attachment, taken from the swapchain.
    // Dynamic rendering needs the format, not a render pass object.
    std::uint32_t colorFormat = 0;
    // Single interleaved vertex binding; stride 0 = no vertex input.
    std::uint32_t vertexStride = 0;
    std::vector<VertexAttribute> vertexAttributes;
    // VkFormat of the depth attachment; 0 = no depth test/attachment. Must
    // match what the frame's rendering info attaches.
    std::uint32_t depthFormat = 0;
    // One push-constant range visible to both stages; 0 = none.
    std::uint32_t pushConstantBytes = 0;
    // Set 0 layout (the bindless DescriptorTable); null = no sets.
    VkDescriptorSetLayout descriptorLayout = nullptr;
    // Alpha blending (src-alpha / one-minus-src-alpha) with depth test but
    // NO depth write — the transparency-pass state. Blended fragments must
    // not occlude each other; opaques drawn first still occlude them.
    bool alphaBlend = false;
    // Occlusion-proxy state: depth TEST only (LESS_OR_EQUAL, so a flat
    // object's zero-extent box coplanar with its own surface still
    // passes), no depth write, color writes fully masked — the pass
    // exists purely for its fragment shader's visibility-buffer stores.
    bool occlusionProxy = false;
};

struct ComputePipelineDesc {
    const Shader* shader = nullptr;
    const char* entryPoint = "main";
    // Set 0 layout (the bindless DescriptorTable); null = no sets.
    VkDescriptorSetLayout descriptorLayout = nullptr;
    // One compute-stage push-constant range; 0 = none.
    std::uint32_t pushConstantBytes = 0;
};

// A graphics or compute pipeline plus the layout it was built with.
// Graphics is dynamic rendering only (no render passes); viewport and
// scissor are dynamic state so a resize never rebuilds the pipeline.
class Pipeline {
public:
    static Result<std::unique_ptr<Pipeline>> createGraphics(const Device& device,
                                                            const GraphicsPipelineDesc& desc);
    static Result<std::unique_ptr<Pipeline>> createCompute(const Device& device,
                                                           const ComputePipelineDesc& desc);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    VkPipeline handle() const { return pipeline_; }
    VkPipelineLayout layout() const { return layout_; }

private:
    Pipeline() = default;

    const Device* device_ = nullptr;
    VkPipelineLayout layout_ = nullptr;
    VkPipeline pipeline_ = nullptr;
};

} // namespace rend::gpu

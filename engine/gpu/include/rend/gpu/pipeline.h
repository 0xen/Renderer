#pragma once

#include "rend/core/result.h"

#include <cstdint>
#include <memory>
#include <vector>

typedef struct VkPipeline_T* VkPipeline;
typedef struct VkPipelineLayout_T* VkPipelineLayout;

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
};

// A graphics pipeline plus the layout it was built with. Dynamic rendering
// only (no render passes); viewport and scissor are dynamic state so a
// resize never rebuilds the pipeline.
class Pipeline {
public:
    static Result<std::unique_ptr<Pipeline>> createGraphics(const Device& device,
                                                            const GraphicsPipelineDesc& desc);
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

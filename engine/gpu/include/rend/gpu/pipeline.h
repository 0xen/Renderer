#pragma once

#include "rend/core/result.h"

#include <cstdint>
#include <memory>

typedef struct VkPipeline_T* VkPipeline;
typedef struct VkPipelineLayout_T* VkPipelineLayout;

namespace rend::gpu {

class Device;
class Shader;

struct GraphicsPipelineDesc {
    const Shader* vertexShader = nullptr;
    const Shader* fragmentShader = nullptr;
    const char* vertexEntryPoint = "main";
    const char* fragmentEntryPoint = "main";
    // VkFormat of the single color attachment, taken from the swapchain.
    // Dynamic rendering needs the format, not a render pass object.
    std::uint32_t colorFormat = 0;
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

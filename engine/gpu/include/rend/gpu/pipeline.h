#pragma once

#include "rend/core/result.h"
#include "rend/gpu/format.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace rend::gpu {

class DescriptorTable;
class Device;
class Shader;

struct VertexAttribute {
    std::uint32_t location = 0;
    Format format = Format::Undefined;
    std::uint32_t offset = 0;
};

struct GraphicsPipelineDesc {
    const Shader* vertexShader = nullptr;
    const Shader* fragmentShader = nullptr;
    const char* vertexEntryPoint = "main";
    const char* fragmentEntryPoint = "main";
    // Format of the single color attachment, taken from the swapchain.
    // Dynamic rendering needs the format, not a render pass object.
    Format colorFormat = Format::Undefined;
    // Multiple render targets (the deferred G-buffer pass): when non-empty
    // this list wins over colorFormat and declares one attachment per
    // entry. Blending/write-mask state is replicated across attachments.
    std::vector<Format> colorFormats;
    // Single interleaved vertex binding; stride 0 = no vertex input.
    std::uint32_t vertexStride = 0;
    std::vector<VertexAttribute> vertexAttributes;
    // Format of the depth attachment; Undefined = no depth test/attachment.
    // Must match what the frame's rendering info attaches.
    Format depthFormat = Format::Undefined;
    // One push-constant range visible to both stages; 0 = none.
    std::uint32_t pushConstantBytes = 0;
    // The bindless DescriptorTable the pipeline binds as set 0; null = no sets.
    const DescriptorTable* descriptorTable = nullptr;
    // Alpha blending (src-alpha / one-minus-src-alpha) with depth test but
    // NO depth write — the transparency-pass state. Blended fragments must
    // not occlude each other; opaques drawn first still occlude them.
    bool alphaBlend = false;
    // Occlusion-proxy state: depth TEST only (LESS_OR_EQUAL, so a flat
    // object's zero-extent box coplanar with its own surface still
    // passes), no depth write, color writes fully masked — the pass
    // exists purely for its fragment shader's visibility-buffer stores.
    bool occlusionProxy = false;
    // Occlusion-proxy DEBUG state (the "show occlusion boxes" overlay):
    // identical depth state to occlusionProxy (LESS_OR_EQUAL test, no
    // write) so the visible fragments match the real proxy pass exactly,
    // but color writes ON with alpha blending — the boxes render as
    // translucent tints instead of being masked out.
    bool occlusionDebug = false;
    // Background state (the sky pass): depth TEST only at LESS_OR_EQUAL —
    // a far-plane fullscreen triangle passes exactly where the cleared
    // depth (1.0) survived the opaques — no depth write, color writes on.
    bool background = false;
    // Depth test AND write fully off while the pass still has a depth
    // attachment (depthFormat must match it): the deferred lighting
    // triangle shades every pixel regardless of stored depth.
    bool disableDepthTest = false;
};

struct ComputePipelineDesc {
    const Shader* shader = nullptr;
    const char* entryPoint = "main";
    // The bindless DescriptorTable the pipeline binds as set 0; null = no sets.
    const DescriptorTable* descriptorTable = nullptr;
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
    virtual ~Pipeline() = default;

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // Which bind point the pipeline belongs to (CommandContext follows it).
    bool isCompute() const { return compute_; }

protected:
    Pipeline() = default;

    bool compute_ = false;
};

} // namespace rend::gpu

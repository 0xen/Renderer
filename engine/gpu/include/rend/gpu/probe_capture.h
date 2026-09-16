#pragma once

#include "rend/core/result.h"
#include "rend/gpu/frame_renderer.h"

#include <array>
#include <cstdint>
#include <memory>

typedef struct VkBuffer_T* VkBuffer;
typedef struct VkDescriptorSet_T* VkDescriptorSet;

namespace rend::gpu {

class Device;
class Image;
class Pipeline;

// One load-time reflection probe capture: the scene's draw stream rendered
// six times into the faces of a fresh cubemap (one synchronous one-shot
// submit, TextureUploader-style), then a blit chain generates the full mip
// pyramid so shaders can pick mips by roughness. The caller provides a
// pipeline built for the face format and pre-writes one camera + light
// region per face; this class executes, it never chooses policy.
struct ProbeCaptureDesc {
    const Buffer* geometry = nullptr; // bound at offset 0 as VB and IB (uint32)
    const DrawIndexedIndirect* draws = nullptr; // CPU draw list (bind pose)
    std::uint32_t drawCount = 0;
    const DescriptorTable* descriptors = nullptr;
    const Pipeline* pipeline = nullptr; // colorFormat must equal `format`
    Format format = Format::Undefined;  // format of the cube faces
    std::uint32_t faceSize = 256;       // square face resolution
    // First camera/light buffer region of the six per-face capture regions;
    // face f is drawn with push constant slot = cameraSlotBase + f.
    std::uint32_t cameraSlotBase = 0;
    // Per-face clear value (the probe's sky stand-in by default; distance
    // captures clear to a huge distance instead).
    std::array<float, 4> clearColor{0.02f, 0.02f, 0.04f, 1.0f};
    // Mip levels to produce via the post-render blit chain; 0 = the full
    // pyramid. 1 skips blitting entirely (formats without blit/filter
    // support, or captures sampled at level 0 only).
    std::uint32_t mipLevels = 0;
};

class ProbeCapture {
public:
    // Renders the probe and returns the mipmapped cube image, ready to
    // sample (SHADER_READ_ONLY across all mips and faces).
    static Result<std::unique_ptr<Image>> render(const Device& device,
                                                 const ProbeCaptureDesc& desc);
};

} // namespace rend::gpu

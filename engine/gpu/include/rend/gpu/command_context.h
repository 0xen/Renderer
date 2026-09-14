#pragma once

#include <array>
#include <cstdint>
#include <vector>

typedef struct VkCommandBuffer_T* VkCommandBuffer;
typedef struct VkBuffer_T* VkBuffer;
typedef struct VkImage_T* VkImage;
typedef struct VkImageView_T* VkImageView;
typedef struct VkDescriptorSet_T* VkDescriptorSet;

namespace rend::gpu {

class Image;
class Pipeline;

// What an image is being used for, from the barrier's point of view. Each
// state maps to a layout plus the stage/access pair that produces or
// consumes it, so a caller transitions between usages without spelling
// out Vulkan's masks. Undefined discards the contents (an image that is
// cleared or fully overwritten every frame starts from it).
enum class ImageState {
    Undefined,
    ColorAttachment, // written by a rendering pass
    DepthAttachment, // depth tested/written by a rendering pass
    ShaderRead,      // sampled or loaded by any shader stage
};

// Pipeline stages a buffer barrier orders between; combine with |.
enum class Stage : std::uint32_t {
    ComputeShader = 1u << 0,
    VertexShader = 1u << 1,
    FragmentShader = 1u << 2,
    DrawIndirect = 1u << 3,
    VertexInput = 1u << 4,
    ColorAttachment = 1u << 5,
};
constexpr Stage operator|(Stage a, Stage b) {
    return static_cast<Stage>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

enum class LoadOp { Clear, Load, DontCare };

struct ColorTarget {
    VkImageView view = nullptr;
    LoadOp load = LoadOp::Clear;
    bool store = true;
    std::array<float, 4> clear{0.0f, 0.0f, 0.0f, 1.0f};
};

struct DepthTarget {
    VkImageView view = nullptr;
    LoadOp load = LoadOp::Clear;
    bool store = false;
    float clear = 1.0f;
};

struct RenderingDesc {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<ColorTarget> colors;
    const DepthTarget* depth = nullptr; // null = no depth attachment
};

// A thin recording surface over one command buffer, handed to frame
// passes (see FramePass in frame_renderer.h). It exposes the verbs a
// custom pass needs — barriers, dynamic rendering, binds, draws,
// dispatches — without the caller including Vulkan headers or owning a
// loader table; raw() is the escape hatch for anything else. Everything
// recorded through it lands in whatever command buffer the frame is
// recording, static or per-frame alike.
class CommandContext {
public:
    explicit CommandContext(VkCommandBuffer cmd) : cmd_(cmd) {}

    VkCommandBuffer raw() const { return cmd_; }

    // Layout + memory barrier for a whole image (mip 0, one layer). The
    // Image overload picks the depth aspect from the image's own format.
    void imageBarrier(const Image& image, ImageState from, ImageState to);
    void imageBarrier(VkImage image, bool depthAspect, ImageState from, ImageState to);
    // Global memory barrier: every write in `from` is visible to every
    // access in `to` (buffers written by one pass, read by the next).
    void memoryBarrier(Stage from, Stage to);

    // Dynamic rendering into the given targets; also sets the viewport and
    // scissor to the full extent. Attachments must already be in their
    // attachment states (imageBarrier). Must be closed with endRendering.
    void beginRendering(const RenderingDesc& desc);
    void endRendering();
    void setViewport(float x, float y, float width, float height);
    void setScissor(std::int32_t x, std::int32_t y, std::uint32_t width, std::uint32_t height);

    // Bind point follows the pipeline's kind (graphics or compute).
    void bindPipeline(const Pipeline& pipeline);
    void bindDescriptorSet(const Pipeline& pipeline, VkDescriptorSet set);
    // Push-constant range at offset 0, all stages the pipeline declared.
    void pushConstants(const Pipeline& pipeline, const void* data, std::uint32_t bytes);

    void bindVertexBuffer(VkBuffer buffer, std::uint64_t offset = 0);
    void bindIndexBuffer(VkBuffer buffer, std::uint64_t offset = 0); // uint32 indices

    void draw(std::uint32_t vertexCount, std::uint32_t instanceCount = 1,
              std::uint32_t firstVertex = 0, std::uint32_t firstInstance = 0);
    void drawIndexed(std::uint32_t indexCount, std::uint32_t instanceCount = 1,
                     std::uint32_t firstIndex = 0, std::int32_t vertexOffset = 0,
                     std::uint32_t firstInstance = 0);
    // Entries are VkDrawIndirectCommand (16 B) / VkDrawIndexedIndirectCommand
    // (20 B, DrawIndexedIndirect in frame_renderer.h); stride 0 = packed.
    void drawIndirect(VkBuffer buffer, std::uint64_t offset, std::uint32_t drawCount,
                      std::uint32_t stride = 0);
    void drawIndexedIndirect(VkBuffer buffer, std::uint64_t offset, std::uint32_t drawCount,
                             std::uint32_t stride = 0);
    void dispatch(std::uint32_t x, std::uint32_t y = 1, std::uint32_t z = 1);

private:
    VkCommandBuffer cmd_ = nullptr;
};

} // namespace rend::gpu

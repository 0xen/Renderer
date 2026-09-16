#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace rend::gpu {

class Buffer;
class DescriptorTable;
class Image;
class Pipeline;

// What an image is being used for, from the barrier's point of view. Each
// state maps to a layout plus the stage/access pair that produces or
// consumes it, so a caller transitions between usages without spelling
// out the API's masks. Undefined discards the contents (an image that is
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

// Explicit synchronization vocabulary for the engine's own passes: a
// one-to-one rendering of the API's stage/access/layout model, so a
// barrier expressed here records exactly what a hand-written one would.
// Frame-pass consumers normally use the simpler ImageState/Stage helpers
// above; these are for recording code that needs precise scopes.
enum class PipelineStage : std::uint32_t {
    None = 0,
    DrawIndirect = 1u << 0,
    VertexAttributeInput = 1u << 1,
    IndexInput = 1u << 2,
    VertexShader = 1u << 3,
    FragmentShader = 1u << 4,
    EarlyFragmentTests = 1u << 5,
    LateFragmentTests = 1u << 6,
    ColorAttachmentOutput = 1u << 7,
    ComputeShader = 1u << 8,
    Clear = 1u << 9,
    AccelerationStructureBuild = 1u << 10,
    AllCommands = 1u << 11,
};
constexpr PipelineStage operator|(PipelineStage a, PipelineStage b) {
    return static_cast<PipelineStage>(static_cast<std::uint32_t>(a) |
                                      static_cast<std::uint32_t>(b));
}

enum class Access : std::uint32_t {
    None = 0,
    IndirectCommandRead = 1u << 0,
    VertexAttributeRead = 1u << 1,
    ShaderRead = 1u << 2,
    ShaderSampledRead = 1u << 3,
    ShaderStorageRead = 1u << 4,
    ShaderStorageWrite = 1u << 5,
    ColorAttachmentRead = 1u << 6,
    ColorAttachmentWrite = 1u << 7,
    DepthStencilRead = 1u << 8,
    DepthStencilWrite = 1u << 9,
    TransferWrite = 1u << 10,
    AccelerationStructureRead = 1u << 11,
    AccelerationStructureWrite = 1u << 12,
};
constexpr Access operator|(Access a, Access b) {
    return static_cast<Access>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

// Image layouts the engine's passes move images between. ShaderReadOnly
// is the generic sampled layout for color AND depth images; Present is
// what the swapchain image must be in when the frame ends.
enum class ImageLayout { Undefined, ColorAttachment, DepthAttachment, ShaderReadOnly, Present };

struct MemoryBarrierDesc {
    PipelineStage srcStage = PipelineStage::None;
    Access srcAccess = Access::None;
    PipelineStage dstStage = PipelineStage::None;
    Access dstAccess = Access::None;
};

struct ImageBarrierDesc {
    const Image* image = nullptr; // mip 0, layer 0; aspect from the format
    ImageLayout oldLayout = ImageLayout::Undefined;
    ImageLayout newLayout = ImageLayout::Undefined;
    PipelineStage srcStage = PipelineStage::None;
    Access srcAccess = Access::None;
    PipelineStage dstStage = PipelineStage::None;
    Access dstAccess = Access::None;
};

enum class LoadOp { Clear, Load, DontCare };

struct ColorTarget {
    const Image* image = nullptr; // rendered through its full view
    LoadOp load = LoadOp::Clear;
    bool store = true;
    std::array<float, 4> clear{0.0f, 0.0f, 0.0f, 1.0f};
};

struct DepthTarget {
    const Image* image = nullptr;
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
// passes (see FramePass in frame_renderer.h) and the overlay recorder. It
// exposes the verbs a custom pass needs — barriers, rendering, binds,
// draws, dispatches — in terms of engine objects only: no graphics-API
// type crosses this header. nativeHandle() is the escape hatch for
// backend-specific integrations (the ImGui backend), which must know
// which backend is active to interpret it.
class CommandContext {
public:
    // Backend-internal: wraps the API command buffer being recorded.
    explicit CommandContext(void* nativeCommandBuffer) : cmd_(nativeCommandBuffer) {}

    // The backend's command buffer (VkCommandBuffer under Vulkan).
    void* nativeHandle() const { return cmd_; }

    // Layout + memory barrier for a whole image (mip 0, one layer); the
    // depth aspect follows the image's format.
    void imageBarrier(const Image& image, ImageState from, ImageState to);
    // Global memory barrier: every write in `from` is visible to every
    // access in `to` (buffers written by one pass, read by the next).
    void memoryBarrier(Stage from, Stage to);

    // Precise barriers (see the vocabulary above): one dependency carrying
    // any number of global memory and image barriers.
    void barrier(std::span<const MemoryBarrierDesc> memory,
                 std::span<const ImageBarrierDesc> images);
    void memoryBarrier(const MemoryBarrierDesc& desc) { barrier({&desc, 1}, {}); }
    void imageBarrier(const ImageBarrierDesc& desc) { barrier({}, {&desc, 1}); }

    // Fills `size` bytes of the buffer at `offset` with a 32-bit value
    // (both multiples of 4); a transfer-stage write.
    void fillBuffer(const Buffer& buffer, std::uint64_t offset, std::uint64_t size,
                    std::uint32_t value);

    // Dynamic rendering into the given targets; also sets the viewport and
    // scissor to the full extent. Attachments must already be in their
    // attachment states (imageBarrier). Must be closed with endRendering.
    void beginRendering(const RenderingDesc& desc);
    void endRendering();
    void setViewport(float x, float y, float width, float height);
    void setScissor(std::int32_t x, std::int32_t y, std::uint32_t width, std::uint32_t height);

    // Bind point follows the pipeline's kind (graphics or compute).
    void bindPipeline(const Pipeline& pipeline);
    void bindDescriptorTable(const Pipeline& pipeline, const DescriptorTable& table);
    // Push-constant range at offset 0, all stages the pipeline declared.
    void pushConstants(const Pipeline& pipeline, const void* data, std::uint32_t bytes);

    void bindVertexBuffer(const Buffer& buffer, std::uint64_t offset = 0);
    void bindIndexBuffer(const Buffer& buffer, std::uint64_t offset = 0); // uint32 indices

    void draw(std::uint32_t vertexCount, std::uint32_t instanceCount = 1,
              std::uint32_t firstVertex = 0, std::uint32_t firstInstance = 0);
    void drawIndexed(std::uint32_t indexCount, std::uint32_t instanceCount = 1,
                     std::uint32_t firstIndex = 0, std::int32_t vertexOffset = 0,
                     std::uint32_t firstInstance = 0);
    // Entries are the API's draw-indirect (16 B) / indexed (20 B,
    // DrawIndexedIndirect in frame_renderer.h) records; stride 0 = packed.
    void drawIndirect(const Buffer& buffer, std::uint64_t offset, std::uint32_t drawCount,
                      std::uint32_t stride = 0);
    void drawIndexedIndirect(const Buffer& buffer, std::uint64_t offset, std::uint32_t drawCount,
                             std::uint32_t stride = 0);
    // GPU-supplied draw count (uint32 at countOffset), capped at maxDrawCount.
    void drawIndexedIndirectCount(const Buffer& buffer, std::uint64_t offset,
                                  const Buffer& count, std::uint64_t countOffset,
                                  std::uint32_t maxDrawCount, std::uint32_t stride = 0);
    void dispatch(std::uint32_t x, std::uint32_t y = 1, std::uint32_t z = 1);

private:
    void* cmd_ = nullptr;
};

} // namespace rend::gpu

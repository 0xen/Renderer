#pragma once

// The Vulkan backend: concrete classes behind the neutral rend::gpu
// interfaces. Private to engine/gpu — nothing outside the layer includes
// this header, so Vulkan types may appear freely here.

#include "rend/gpu/acceleration_structure.h"
#include "rend/gpu/buffer.h"
#include "rend/gpu/command_context.h"
#include "rend/gpu/descriptor_table.h"
#include "rend/gpu/device.h"
#include "rend/gpu/frame_renderer.h"
#include "rend/gpu/image.h"
#include "rend/gpu/instance.h"
#include "rend/gpu/memory_tracker.h"
#include "rend/gpu/pipeline.h"
#include "rend/gpu/shader.h"
#include "rend/gpu/swapchain.h"
#include "rend/gpu/texture_uploader.h"
#include "rend/gpu/transfer.h"

#include <volk.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rend::gpu {

class VulkanInstance final : public Instance {
public:
    static Result<std::unique_ptr<Instance>> create(const InstanceDesc& desc);
    ~VulkanInstance() override;

    VkInstance handle() const { return instance_; }
    void* nativeHandle() const override { return instance_; }
    bool validationEnabled() const override { return messenger_ != nullptr; }
    std::uint32_t apiVersion() const override { return apiVersion_; }

private:
    VulkanInstance() : Instance(Api::Vulkan) {}

    VkInstance instance_ = nullptr;
    VkDebugUtilsMessengerEXT messenger_ = nullptr;
    std::uint32_t apiVersion_ = 0;
};

struct QueueInfo {
    VkQueue queue = nullptr;
    std::uint32_t familyIndex = ~0u;
    bool valid() const { return queue != nullptr; }
};

class VulkanDevice final : public Device {
public:
    static Result<std::unique_ptr<Device>> create(const Instance& instance,
                                                  const FeatureSet& features);
    ~VulkanDevice() override;

    VkDevice handle() const { return device_; }
    VkPhysicalDevice physicalDevice() const { return physical_; }
    // Graphics queue also handles compute + present.
    const QueueInfo& graphicsQueue() const { return graphics_; }
    // Dedicated transfer-only family when the hardware has one (async
    // streaming/defrag); falls back to the graphics queue otherwise.
    const QueueInfo& transferQueue() const { return transfer_; }

    bool hasDedicatedTransfer() const override {
        return transfer_.familyIndex != graphics_.familyIndex;
    }
    void waitIdle() const override;
    void* nativeHandle() const override { return device_; }
    void* nativePhysicalDevice() const override { return physical_; }
    void* nativeGraphicsQueue() const override { return graphics_.queue; }
    std::uint32_t graphicsQueueFamily() const override { return graphics_.familyIndex; }

private:
    VulkanDevice() : Device(Api::Vulkan) {}

    VkPhysicalDevice physical_ = nullptr;
    VkDevice device_ = nullptr;
    QueueInfo graphics_;
    QueueInfo transfer_;
};

class VulkanBuffer final : public Buffer {
public:
    static Result<std::unique_ptr<Buffer>> create(const Device& device, const BufferDesc& desc);
    ~VulkanBuffer() override;

    VkBuffer handle() const { return buffer_; }

private:
    VulkanBuffer() = default;

    const VulkanDevice* device_ = nullptr;
    VkBuffer buffer_ = nullptr;
    VkDeviceMemory memory_ = nullptr;
    // What the allocation actually cost (alignment-padded) — what the
    // MemoryTracker was told, so the destructor releases the same figure.
    std::uint64_t allocatedBytes_ = 0;
    MemoryTracker::Kind trackKind_ = MemoryTracker::Kind::DeviceBuffer;
};

class VulkanImage final : public Image {
public:
    static Result<std::unique_ptr<Image>> create(const Device& device, const ImageDesc& desc);
    // A non-owning view of an image the backend does not allocate itself
    // (swapchain images). Destroys nothing.
    static std::unique_ptr<Image> wrapExternal(VkImage image, VkImageView view, Format format,
                                               std::uint32_t width, std::uint32_t height);
    ~VulkanImage() override;

    VkImage handle() const { return image_; }
    VkImageView view() const { return view_; }
    // Cube images only: 2D render view of one face's mip 0.
    VkImageView faceView(std::uint32_t face) const { return faceViews_[face]; }

private:
    VulkanImage() = default;

    const VulkanDevice* device_ = nullptr;
    VkImage image_ = nullptr;
    VkDeviceMemory memory_ = nullptr;
    VkImageView view_ = nullptr;
    std::array<VkImageView, 6> faceViews_{};
    std::uint64_t allocatedBytes_ = 0;
    bool owned_ = true;
};

class VulkanShader final : public Shader {
public:
    static Result<std::unique_ptr<Shader>> createFromFile(const Device& device,
                                                          const std::filesystem::path& path);
    ~VulkanShader() override;

    VkShaderModule handle() const { return module_; }

private:
    VulkanShader() = default;

    const VulkanDevice* device_ = nullptr;
    VkShaderModule module_ = nullptr;
};

class VulkanPipeline final : public Pipeline {
public:
    static Result<std::unique_ptr<Pipeline>> createGraphics(const Device& device,
                                                            const GraphicsPipelineDesc& desc);
    static Result<std::unique_ptr<Pipeline>> createCompute(const Device& device,
                                                           const ComputePipelineDesc& desc);
    ~VulkanPipeline() override;

    VkPipeline handle() const { return pipeline_; }
    VkPipelineLayout layout() const { return layout_; }

private:
    VulkanPipeline() = default;

    const VulkanDevice* device_ = nullptr;
    VkPipelineLayout layout_ = nullptr;
    VkPipeline pipeline_ = nullptr;
};

class VulkanDescriptorTable final : public DescriptorTable {
public:
    static Result<std::unique_ptr<DescriptorTable>> create(const Device& device,
                                                           const DescriptorTableDesc& desc);
    ~VulkanDescriptorTable() override;

    VkDescriptorSetLayout layout() const { return layout_; }
    VkDescriptorSet set() const { return set_; }

    void writeObjectBuffer(const Buffer& buffer, std::uint64_t range) override;
    void writeTexture(std::uint32_t index, const Image& image) override;
    void writeStorageBuffer(std::uint32_t binding, const Buffer& buffer,
                            std::uint64_t range) override;
    void writeShadowMap(std::uint32_t cascade, const Image& image) override;
    void writeProbe(const Image& image) override;
    void writePointShadowMap(std::uint32_t index, const Image& image) override;
    void writeSampledImage(std::uint32_t binding, std::uint32_t index,
                           const Image& image) override;
    void writeAccelerationStructure(const AccelerationStructure& tlas) override;

private:
    VulkanDescriptorTable() = default;

    const VulkanDevice* device_ = nullptr;
    VkDescriptorPool pool_ = nullptr;
    VkDescriptorSetLayout layout_ = nullptr;
    VkDescriptorSet set_ = nullptr;
    VkSampler sampler_ = nullptr;
    VkSampler shadowSampler_ = nullptr; // comparison (PCF) sampler, binding 9
};

class VulkanCommandContext final : public CommandContext {
public:
    explicit VulkanCommandContext(VkCommandBuffer cmd) : cmd_(cmd) {}

    void* nativeHandle() const override { return cmd_; }
    void imageBarrier(const Image& image, ImageState from, ImageState to) override;
    void memoryBarrier(Stage from, Stage to) override;
    void barrier(std::span<const MemoryBarrierDesc> memory,
                 std::span<const ImageBarrierDesc> images) override;
    void fillBuffer(const Buffer& buffer, std::uint64_t offset, std::uint64_t size,
                    std::uint32_t value) override;
    void beginRendering(const RenderingDesc& desc) override;
    void endRendering() override;
    void setViewport(float x, float y, float width, float height) override;
    void setScissor(std::int32_t x, std::int32_t y, std::uint32_t width,
                    std::uint32_t height) override;
    void bindPipeline(const Pipeline& pipeline) override;
    void bindDescriptorTable(const Pipeline& pipeline, const DescriptorTable& table) override;
    void pushConstants(const Pipeline& pipeline, const void* data, std::uint32_t bytes) override;
    void bindVertexBuffer(const Buffer& buffer, std::uint64_t offset) override;
    void bindIndexBuffer(const Buffer& buffer, std::uint64_t offset) override;
    void draw(std::uint32_t vertexCount, std::uint32_t instanceCount, std::uint32_t firstVertex,
              std::uint32_t firstInstance) override;
    void drawIndexed(std::uint32_t indexCount, std::uint32_t instanceCount,
                     std::uint32_t firstIndex, std::int32_t vertexOffset,
                     std::uint32_t firstInstance) override;
    void drawIndirect(const Buffer& buffer, std::uint64_t offset, std::uint32_t drawCount,
                      std::uint32_t stride) override;
    void drawIndexedIndirect(const Buffer& buffer, std::uint64_t offset, std::uint32_t drawCount,
                             std::uint32_t stride) override;
    void drawIndexedIndirectCount(const Buffer& buffer, std::uint64_t offset, const Buffer& count,
                                  std::uint64_t countOffset, std::uint32_t maxDrawCount,
                                  std::uint32_t stride) override;
    void dispatch(std::uint32_t x, std::uint32_t y, std::uint32_t z) override;

private:
    VkCommandBuffer cmd_ = nullptr;
};

class VulkanSwapchain final : public Swapchain {
public:
    static Result<std::unique_ptr<Swapchain>> create(const Instance& instance,
                                                     const Device& device,
                                                     const SwapchainDesc& desc);
    ~VulkanSwapchain() override;

    Result<void> recreate(std::uint32_t width, std::uint32_t height) override;

    VkSwapchainKHR handle() const { return swapchain_; }
    const std::vector<VkImage>& images() const { return images_; }
    const std::vector<VkImageView>& imageViews() const { return views_; }

private:
    VulkanSwapchain() = default;
    Result<void> build(std::uint32_t width, std::uint32_t height, VkSwapchainKHR old);
    void destroyViews();

    const VulkanInstance* instance_ = nullptr;
    const VulkanDevice* device_ = nullptr;
    VkSurfaceKHR surface_ = nullptr;
    VkSwapchainKHR swapchain_ = nullptr;
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
};

class VulkanTransferContext final : public TransferContext {
public:
    static Result<std::unique_ptr<TransferContext>> create(const Device& device);
    ~VulkanTransferContext() override;

    Result<void> stage(const Buffer& dst, std::uint64_t dstOffset, const void* data,
                       std::uint64_t size) override;
    Result<void> flush() override;
    std::uint64_t pendingBytes() const override { return stagingUsed_; }

private:
    VulkanTransferContext() = default;
    Result<void> ensureStagingCapacity(std::uint64_t required);

    struct PendingCopy {
        const Buffer* dst = nullptr;
        std::uint64_t srcOffset = 0;
        std::uint64_t dstOffset = 0;
        std::uint64_t size = 0;
    };

    const VulkanDevice* device_ = nullptr;
    VkCommandPool pool_ = nullptr;
    VkCommandBuffer cmd_ = nullptr;
    VkFence fence_ = nullptr;
    std::unique_ptr<Buffer> staging_;
    std::uint64_t stagingUsed_ = 0;
    std::vector<PendingCopy> pending_;
};

class VulkanTextureUploader final : public TextureUploader {
public:
    static Result<std::unique_ptr<TextureUploader>> create(const Device& device);
    ~VulkanTextureUploader() override;

    Result<std::unique_ptr<Image>> upload(std::uint32_t width, std::uint32_t height,
                                          const void* rgba8, bool srgb) override;
    Result<std::unique_ptr<Image>> uploadCompressed(Format format, const CompressedMip* mips,
                                                    std::uint32_t mipCount, const void* bytes,
                                                    std::uint64_t byteSize) override;

private:
    VulkanTextureUploader() = default;
    Result<void> ensureStagingCapacity(std::uint64_t required);

    const VulkanDevice* device_ = nullptr;
    VkCommandPool pool_ = nullptr;
    VkCommandBuffer cmd_ = nullptr;
    VkFence fence_ = nullptr;
    std::unique_ptr<Buffer> staging_;
};

class VulkanAccelerationStructure final : public AccelerationStructure {
public:
    static Result<std::unique_ptr<AccelerationStructure>> buildBottomLevel(
        const Device& device, std::span<const TriangleGeometry> geometries, bool allowUpdate);
    static Result<std::unique_ptr<AccelerationStructure>> buildTopLevel(
        const Device& device, std::span<const Instance> instances, bool allowUpdate);
    static Result<std::unique_ptr<AccelerationStructure>> buildTopLevelDynamic(
        const Device& device, std::span<const Instance> initial, std::uint32_t capacity,
        std::uint32_t slotCount);
    ~VulkanAccelerationStructure() override;

    VkAccelerationStructureKHR handle() const { return as_; }

    void writeInstances(std::uint32_t slot, std::span<const Instance> instances) override;
    void recordRebuild(CommandContext& cmd, std::uint32_t slot) const override;
    void recordRefit(CommandContext& cmd,
                     std::span<const TriangleGeometry> geometries) const override;
    void recordRefit(CommandContext& cmd) const override;

private:
    VulkanAccelerationStructure() = default;

    const VulkanDevice* device_ = nullptr;
    VkAccelerationStructureKHR as_ = nullptr;
    std::unique_ptr<Buffer> storage_;
    // Refit state (allowUpdate builds only): scratch sized for UPDATE mode,
    // and for a TLAS the live instance buffer the update rereads.
    std::unique_ptr<Buffer> updateScratch_;
    std::unique_ptr<Buffer> instances_;
    std::uint32_t instanceCount_ = 0;
    // Dynamic-TLAS state: BUILD-mode scratch kept for per-frame rebuilds
    // plus the per-slot region geometry of the instance buffer.
    std::unique_ptr<Buffer> buildScratch_;
    std::uint32_t capacity_ = 0;
    std::uint32_t slotCount_ = 0;
};

// The Vulkan frame loop: command pool + per-slot command buffers, fences,
// per-image semaphores, acquire/submit/present, static recordings. The
// passes themselves are the neutral FrameRenderer::recordFrame.
class VulkanFrameRenderer final : public FrameRenderer {
public:
    static Result<std::unique_ptr<FrameRenderer>> create(const Device& device,
                                                         Swapchain& swapchain);
    ~VulkanFrameRenderer() override;

    Result<void> drawFrame(const DrawBatch* batch) override;
    Result<void> waitFrameSlot() override;
    void waitIdle() const override;

private:
    VulkanFrameRenderer() = default;

    Result<void> createSyncObjects();
    Result<void> createImageSemaphores();
    void destroyImageSemaphores();
    Result<void> waitForFence(VkFence fence, const char* what) const;
    Result<void> record(VkCommandBuffer cmd, std::uint32_t imageIndex, std::uint32_t slot,
                        const DrawBatch* batch, bool reusable) const;
    Result<void> recordOverlay(VkCommandBuffer cmd, std::uint32_t imageIndex) const;
    Result<void> prerecordStatic(const DrawBatch* batch);
    void invalidateStatic() override;
    Result<void> onSwapchainRecreated(std::uint32_t previousImageCount) override;

    struct FrameData {
        VkCommandBuffer commandBuffer = nullptr;
        // The per-frame UI/present-transition tail after the scene buffer.
        VkCommandBuffer overlayCommandBuffer = nullptr;
        VkSemaphore imageAvailable = nullptr;
        VkFence inFlight = nullptr;
    };

    const VulkanDevice* vkDevice_ = nullptr;
    VulkanSwapchain* vkSwapchain_ = nullptr;
    VkCommandPool commandPool_ = nullptr;
    std::array<FrameData, kFramesInFlight> frames_{};
    // One per swapchain image, not per frame in flight: presentation may
    // still be reading an image's semaphore when its frame slot comes round.
    std::vector<VkSemaphore> renderFinished_;
    // Static-mode recordings, indexed [slot * imageCount + imageIndex];
    // empty while invalid. A (slot, image) pair is never in flight twice,
    // so the buffers need no simultaneous-use flag.
    std::vector<VkCommandBuffer> staticBuffers_;
};

// Downcasts for the backend's own code: every object created through a
// Vulkan device IS one of these.
inline const VulkanInstance& vk(const Instance& i) { return static_cast<const VulkanInstance&>(i); }
inline const VulkanDevice& vk(const Device& d) { return static_cast<const VulkanDevice&>(d); }
inline const VulkanSwapchain& vk(const Swapchain& s) { return static_cast<const VulkanSwapchain&>(s); }
inline VulkanSwapchain& vk(Swapchain& s) { return static_cast<VulkanSwapchain&>(s); }
inline const VulkanBuffer& vk(const Buffer& b) { return static_cast<const VulkanBuffer&>(b); }
inline const VulkanImage& vk(const Image& i) { return static_cast<const VulkanImage&>(i); }
inline const VulkanShader& vk(const Shader& s) { return static_cast<const VulkanShader&>(s); }
inline const VulkanPipeline& vk(const Pipeline& p) { return static_cast<const VulkanPipeline&>(p); }
inline const VulkanDescriptorTable& vk(const DescriptorTable& t) {
    return static_cast<const VulkanDescriptorTable&>(t);
}
inline const VulkanAccelerationStructure& vk(const AccelerationStructure& a) {
    return static_cast<const VulkanAccelerationStructure&>(a);
}
inline VkCommandBuffer vk(CommandContext& c) { return static_cast<VkCommandBuffer>(c.nativeHandle()); }

} // namespace rend::gpu

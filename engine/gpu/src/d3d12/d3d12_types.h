#pragma once

// The D3D12 backend: concrete classes behind the neutral rend::gpu
// interfaces. Private to engine/gpu — nothing outside the layer includes
// this header, so D3D12/DXGI types may appear freely here.
//
// Slice 2 built the skeleton (instance, device on the display adapter,
// buffers, images, barriers + clears, the DirectComposition swapchain
// with premultiplied alpha, the frame loop). Slice 3 adds the scene
// path: DXIL shaders, root signature + PSOs, the bindless descriptor
// heap, every CommandContext verb (with buffer resource-state tracking),
// uploads. Acceleration structures (DXR) remain slice 4 — the device
// reports the ray-tracing features as unavailable so the viewer takes
// the raster path.

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

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <d3d12.h>
#include <dcomp.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace rend::gpu {

using Microsoft::WRL::ComPtr;

class D3D12Buffer;
class D3D12DescriptorTable;
class D3D12Pipeline;

// Format vocabulary -> DXGI. Undefined/unknown map to DXGI_FORMAT_UNKNOWN.
DXGI_FORMAT toDxgi(Format format);

class D3D12Instance final : public Instance {
public:
    static Result<std::unique_ptr<Instance>> create(const InstanceDesc& desc);
    ~D3D12Instance() override = default;

    IDXGIFactory6* factory() const { return factory_.Get(); }
    void* nativeHandle() const override { return factory_.Get(); }
    bool validationEnabled() const override { return debug_; }
    std::uint32_t apiVersion() const override { return 12; }

private:
    D3D12Instance() : Instance(Api::D3D12) {}

    ComPtr<IDXGIFactory6> factory_;
    bool debug_ = false;
};

class D3D12Device final : public Device {
public:
    static Result<std::unique_ptr<Device>> create(const Instance& instance,
                                                  const FeatureSet& features);
    ~D3D12Device() override;

    ID3D12Device* handle() const { return device_.Get(); }
    IDXGIAdapter1* adapter() const { return adapter_.Get(); }
    // Direct queue: graphics + compute + present.
    ID3D12CommandQueue* graphicsQueue() const { return graphics_.Get(); }
    // Copy queue for uploads (always a distinct engine on D3D12).
    ID3D12CommandQueue* copyQueue() const { return copy_.Get(); }

    bool hasDedicatedTransfer() const override { return true; }
    void waitIdle() const override;
    void* nativeHandle() const override { return device_.Get(); }
    void* nativePhysicalDevice() const override { return adapter_.Get(); }
    void* nativeGraphicsQueue() const override { return graphics_.Get(); }
    std::uint32_t graphicsQueueFamily() const override { return 0; }

    // Forwards everything the debug layer stored since the last call to
    // the log as "[d3d12]" lines (the validation grep the benches use).
    // No-op without the debug layer.
    void drainDebugMessages() const;

    std::uint32_t rtvDescriptorSize() const { return rtvSize_; }
    std::uint32_t dsvDescriptorSize() const { return dsvSize_; }
    std::uint32_t resourceDescriptorSize() const { return cbvSrvUavSize_; }
    std::uint32_t samplerDescriptorSize() const { return samplerSize_; }

    // CPU-visible heap that allows every buffer state (a CUSTOM heap with
    // the UPLOAD heap's page properties): the viewer's per-slot indirect,
    // count and instance-row buffers are host-written AND GPU-written,
    // which an UPLOAD heap (stuck in GENERIC_READ) cannot express.
    const D3D12_HEAP_PROPERTIES& hostHeapProperties() const { return hostHeap_; }

    // Buffer resource-state tracking. Buffers decay to COMMON at the end
    // of every ExecuteCommandLists and promote implicitly from COMMON on
    // first use, so the recorder resets every tracked buffer to COMMON
    // when it starts a command list and names explicit before-states
    // only for buffers it already moved within that list. Only buffers
    // that can be UAVs are tracked (the ones shaders/indirect draws use).
    void registerBuffer(const D3D12Buffer* buffer) const;
    void unregisterBuffer(const D3D12Buffer* buffer) const;
    const std::vector<const D3D12Buffer*>& trackedBuffers() const { return tracked_; }
    void resetBufferStates() const;

    // One MiB of zeros in an upload heap (always readable as a copy
    // source): fillBuffer(0) is a CopyBufferRegion from it.
    ID3D12Resource* zeroBuffer() const { return zero_.Get(); }
    static constexpr std::uint64_t kZeroBytes = 1u << 20;

    // Command signature for non-indexed indirect draws of the given
    // stride (no root-signature arguments, so the device owns them).
    Result<ID3D12CommandSignature*> drawSignature(std::uint32_t stride) const;

private:
    D3D12Device() : Device(Api::D3D12) {}

    ComPtr<IDXGIAdapter1> adapter_;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> graphics_;
    ComPtr<ID3D12CommandQueue> copy_;
    ComPtr<ID3D12InfoQueue> infoQueue_;
    // waitIdle: one fence signalled through both queues.
    ComPtr<ID3D12Fence> idleFence_;
    mutable std::uint64_t idleValue_ = 0;
    HANDLE idleEvent_ = nullptr;
    std::uint32_t rtvSize_ = 0;
    std::uint32_t dsvSize_ = 0;
    std::uint32_t cbvSrvUavSize_ = 0;
    std::uint32_t samplerSize_ = 0;
    D3D12_HEAP_PROPERTIES hostHeap_{};
    ComPtr<ID3D12Resource> zero_;
    mutable std::vector<const D3D12Buffer*> tracked_;
    mutable std::vector<std::pair<std::uint32_t, ComPtr<ID3D12CommandSignature>>> drawSignatures_;
};

class D3D12Buffer final : public Buffer {
public:
    static Result<std::unique_ptr<Buffer>> create(const Device& device, const BufferDesc& desc);
    ~D3D12Buffer() override;

    ID3D12Resource* resource() const { return resource_.Get(); }
    D3D12_GPU_VIRTUAL_ADDRESS gpuAddress() const { return resource_->GetGPUVirtualAddress(); }
    // Created with ALLOW_UNORDERED_ACCESS (storage usage): tracked by the
    // device for resource-state bookkeeping.
    bool tracked() const { return tracked_; }
    // Bound at a UAV binding by some descriptor table: shader access needs
    // UNORDERED_ACCESS; read-only buffers (SRV bindings only) sit in the
    // shader-resource states instead.
    bool uavBound() const { return uavBound_; }
    void markUavBound() const { uavBound_ = true; }

    // Current resource state within the command list being recorded
    // (COMMON between lists — see D3D12Device::resetBufferStates).
    D3D12_RESOURCE_STATES state() const { return state_; }
    void setState(D3D12_RESOURCE_STATES state) const { state_ = state; }

private:
    D3D12Buffer() = default;

    const D3D12Device* device_ = nullptr;
    ComPtr<ID3D12Resource> resource_;
    mutable D3D12_RESOURCE_STATES state_ = D3D12_RESOURCE_STATE_COMMON;
    std::uint64_t allocatedBytes_ = 0;
    MemoryTracker::Kind trackKind_ = MemoryTracker::Kind::DeviceBuffer;
    bool tracked_ = false;
    mutable bool uavBound_ = false;
};

class D3D12Image final : public Image {
public:
    static Result<std::unique_ptr<Image>> create(const Device& device, const ImageDesc& desc);
    // A non-owning view of a swapchain buffer: the resource stays owned by
    // the swapchain (which also owns the RTV heap the handle points into).
    static std::unique_ptr<Image> wrapExternal(ID3D12Resource* resource,
                                               D3D12_CPU_DESCRIPTOR_HANDLE rtv, Format format,
                                               std::uint32_t width, std::uint32_t height);
    ~D3D12Image() override;

    ID3D12Resource* resource() const { return resource_.Get(); }
    D3D12_CPU_DESCRIPTOR_HANDLE rtv() const { return rtv_; }
    D3D12_CPU_DESCRIPTOR_HANDLE dsv() const { return dsv_; }
    bool hasRtv() const { return rtv_.ptr != 0; }
    bool hasDsv() const { return dsv_.ptr != 0; }
    // The full-mip shader-resource view the descriptor table writes:
    // R32_FLOAT over a typeless depth resource, the image format else.
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc() const;

    // D3D12 barriers name the before-state explicitly, so the image
    // tracks where it currently sits (single-queue recording order).
    D3D12_RESOURCE_STATES state() const { return state_; }
    void setState(D3D12_RESOURCE_STATES state) const { state_ = state; }

private:
    D3D12Image() = default;

    ComPtr<ID3D12Resource> resource_;
    ComPtr<ID3D12DescriptorHeap> rtvHeap_; // owned views (created images only)
    ComPtr<ID3D12DescriptorHeap> dsvHeap_;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv_{};
    D3D12_CPU_DESCRIPTOR_HANDLE dsv_{};
    DXGI_FORMAT srvFormat_ = DXGI_FORMAT_UNKNOWN;
    mutable D3D12_RESOURCE_STATES state_ = D3D12_RESOURCE_STATE_COMMON;
    std::uint64_t allocatedBytes_ = 0;
    bool owned_ = true;
};

// One DXIL container loaded from disk. The viewer names its shaders by
// the SPIR-V artifact (<name>.<stage>.spv); the D3D12 build of the same
// stage sits beside it as <name>.<stage>.dxil.
class D3D12Shader final : public Shader {
public:
    static Result<std::unique_ptr<Shader>> createFromFile(const Device& device,
                                                          const std::filesystem::path& path);
    ~D3D12Shader() override = default;

    D3D12_SHADER_BYTECODE bytecode() const {
        return D3D12_SHADER_BYTECODE{code_.data(), code_.size()};
    }

private:
    D3D12Shader() = default;

    std::vector<std::byte> code_;
};

// The bindless table as one shader-visible CBV_SRV_UAV heap plus a two-
// entry sampler heap, and the root signature every pipeline binding the
// table shares: [0] root constants b0 (the push block, kPushDwords),
// [1] root constant b1 (the draw's base instance — D3D12's SV_InstanceID
// excludes it, see assets/shaders/backend.hlsli), [2] one descriptor
// table with a range per binding (binding N = register space N, slot =
// the range's offset from the table start), [3] the sampler table.
// Shader-written storage buffers (and the read-only aliases of the same
// buffers) are UAV ranges, read-only buffers and images SRV ranges — see
// assets/shaders/backend.hlsli for the split and why it matters.
class D3D12DescriptorTable final : public DescriptorTable {
public:
    // What a heap slot holds: a UAV over a shader-written buffer, an SRV
    // over a read-only buffer, or an SRV over an image (or the TLAS).
    enum class Kind { UavBuffer, SrvBuffer, SrvImage };
    static constexpr std::uint32_t kPushDwords = 16;
    // Each indirect draw record sets root parameter 1 from its leading
    // word before the draw arguments.
    static constexpr std::uint32_t kIndirectStride = sizeof(DrawIndexedIndirect);

    static Result<std::unique_ptr<DescriptorTable>> create(const Device& device,
                                                           const DescriptorTableDesc& desc);
    ~D3D12DescriptorTable() override = default;

    void writeObjectBuffer(const Buffer& buffer, std::uint64_t range = 0) override;
    void writeTexture(std::uint32_t index, const Image& image) override;
    void writeStorageBuffer(std::uint32_t binding, const Buffer& buffer,
                            std::uint64_t range = 0) override;
    void writeShadowMap(std::uint32_t cascade, const Image& image) override;
    void writeProbe(const Image& image) override;
    void writePointShadowMap(std::uint32_t index, const Image& image) override;
    void writeSampledImage(std::uint32_t binding, std::uint32_t index,
                           const Image& image) override;
    void writeAccelerationStructure(const AccelerationStructure& tlas) override;

    ID3D12RootSignature* rootSignature() const { return rootSignature_.Get(); }
    ID3D12DescriptorHeap* resourceHeap() const { return resourceHeap_.Get(); }
    ID3D12DescriptorHeap* samplerHeap() const { return samplerHeap_.Get(); }
    D3D12_GPU_DESCRIPTOR_HANDLE resourceTableStart() const {
        return resourceHeap_->GetGPUDescriptorHandleForHeapStart();
    }
    D3D12_GPU_DESCRIPTOR_HANDLE samplerTableStart() const {
        return samplerHeap_->GetGPUDescriptorHandleForHeapStart();
    }
    // {root constant b1, DrawIndexedInstanced} over DrawIndexedIndirect
    // records (kIndirectStride bytes each).
    ID3D12CommandSignature* drawIndexedSignature() const { return drawIndexedSignature_.Get(); }

private:
    struct Slot {
        std::uint32_t binding = 0;
        Kind kind = Kind::SrvImage;
        bool cube = false;  // TextureCube SRV
        bool depth = false; // sampled depth (R32_FLOAT over typeless)
        std::uint32_t count = 1;
        std::uint32_t stride = 0; // structured stride; 0 = raw (ByteAddressBuffer)
        std::uint32_t first = 0;  // heap slot of element 0
    };

    D3D12DescriptorTable() = default;
    const Slot* slot(std::uint32_t binding) const;
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle(std::uint32_t index) const;
    void writeBufferView(const Slot& slot, const Buffer& buffer, std::uint64_t range);
    void writeImageView(const Slot& slot, std::uint32_t index, const Image& image);

    const D3D12Device* device_ = nullptr;
    std::vector<Slot> slots_;
    ComPtr<ID3D12DescriptorHeap> resourceHeap_;
    ComPtr<ID3D12DescriptorHeap> samplerHeap_;
    ComPtr<ID3D12RootSignature> rootSignature_;
    ComPtr<ID3D12CommandSignature> drawIndexedSignature_;
};

// A PSO plus the root signature it was built against (the descriptor
// table's) and the per-vertex stride the vertex-buffer view needs.
class D3D12Pipeline final : public Pipeline {
public:
    static Result<std::unique_ptr<Pipeline>> createGraphics(const Device& device,
                                                            const GraphicsPipelineDesc& desc);
    static Result<std::unique_ptr<Pipeline>> createCompute(const Device& device,
                                                           const ComputePipelineDesc& desc);
    ~D3D12Pipeline() override = default;

    ID3D12PipelineState* pso() const { return pso_.Get(); }
    ID3D12RootSignature* rootSignature() const { return rootSignature_; }
    std::uint32_t vertexStride() const { return vertexStride_; }

private:
    D3D12Pipeline() = default;

    ComPtr<ID3D12PipelineState> pso_;
    ID3D12RootSignature* rootSignature_ = nullptr; // owned by the table
    std::uint32_t vertexStride_ = 0;
};

// Records into one graphics command list: barriers (image transitions
// from the tracked state; global memory barriers become UAV barriers),
// render-target setup + clears, viewport/scissor, and — with buffer
// state tracking applied at every draw/dispatch — binds, root constants,
// direct and ExecuteIndirect draws, dispatches and zero fills.
class D3D12CommandContext final : public CommandContext {
public:
    D3D12CommandContext(ID3D12GraphicsCommandList* list, const D3D12Device& device)
        : list_(list), device_(&device) {}

    void* nativeHandle() const override { return list_; }
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
    void transition(const Image& image, D3D12_RESOURCE_STATES to);
    // Moves every tracked buffer into the state the next command needs:
    // the bound vertex/index buffer to VERTEX_AND_CONSTANT|INDEX, the
    // indirect argument/count buffers to INDIRECT_ARGUMENT, UAV-bound
    // buffers to UNORDERED_ACCESS and read-only ones to the shader-
    // resource states. Explicit transitions even from COMMON (buffers
    // decay to COMMON between lists, so the before-state is exact).
    void applyBufferStates(bool graphics, const D3D12Buffer* indirectArgs,
                           const D3D12Buffer* indirectCount);
    void bufferTransition(const D3D12Buffer& buffer, D3D12_RESOURCE_STATES to,
                          std::vector<D3D12_RESOURCE_BARRIER>& out);
    void setBaseInstance(std::uint32_t firstInstance);

    ID3D12GraphicsCommandList* list_ = nullptr;
    const D3D12Device* device_ = nullptr;
    const D3D12Pipeline* pipeline_ = nullptr;
    const D3D12DescriptorTable* table_ = nullptr;
    ID3D12RootSignature* graphicsRootSignature_ = nullptr;
    ID3D12RootSignature* computeRootSignature_ = nullptr;
    const D3D12Buffer* vertexBuffer_ = nullptr;
    const D3D12Buffer* indexBuffer_ = nullptr;
};

// The presentable surface. Transparent targets get a composition
// swapchain (BGRA8, FLIP_SEQUENTIAL, DXGI_ALPHA_MODE_PREMULTIPLIED) shown
// through a DirectComposition visual on the window — the only D3D path
// that composites per-pixel alpha over the desktop. Opaque targets get a
// plain HWND flip swapchain. SwapchainDesc::nativeSurface is the HWND.
// Both present through a UNORM BGRA8 surface (composition rejects sRGB)
// viewed through sRGB render-target views, so the reported image format
// is B8G8R8A8Srgb and pipelines targeting it encode sRGB in hardware.
class D3D12Swapchain final : public Swapchain {
public:
    static Result<std::unique_ptr<Swapchain>> create(const Instance& instance,
                                                     const Device& device,
                                                     const SwapchainDesc& desc);
    ~D3D12Swapchain() override;

    Result<void> recreate(std::uint32_t width, std::uint32_t height) override;

    IDXGISwapChain3* handle() const { return swapchain_.Get(); }
    std::uint32_t currentBackBufferIndex() const { return swapchain_->GetCurrentBackBufferIndex(); }
    bool transparent() const { return transparent_; }
    // HWND chains created with ALLOW_TEARING when DXGI offers it: a
    // vsync-off Present may then tear instead of being throttled to the
    // compositor's refresh (composition chains never tear).
    bool tearing() const { return tearing_; }
    // Frame-latency waitable object (max latency = kFramesInFlight): the
    // frame loop blocks on it before recording so presentation paces the
    // CPU instead of Present() stalling with a full queue.
    HANDLE frameLatencyWaitable() const { return waitable_; }

private:
    D3D12Swapchain() = default;
    Result<void> build(std::uint32_t width, std::uint32_t height);
    Result<void> acquireBuffers();
    void releaseBuffers();

    const D3D12Instance* instance_ = nullptr;
    const D3D12Device* device_ = nullptr;
    HWND hwnd_ = nullptr;
    ComPtr<IDXGISwapChain3> swapchain_;
    ComPtr<IDCompositionDevice> compositionDevice_;
    ComPtr<IDCompositionTarget> compositionTarget_;
    ComPtr<IDCompositionVisual> compositionVisual_;
    ComPtr<ID3D12DescriptorHeap> rtvHeap_;
    std::vector<ComPtr<ID3D12Resource>> buffers_;
    std::uint32_t bufferCount_ = 0;
    bool tearing_ = false;
    HANDLE waitable_ = nullptr; // owned by the swapchain object
};

// Batched buffer uploads on the copy queue: stage() appends to an upload-
// heap staging buffer, flush() records one CopyBufferRegion per pending
// copy and blocks on a fence. Destination buffers sit in COMMON between
// frames, which the copy queue accepts as is.
class D3D12TransferContext final : public TransferContext {
public:
    static Result<std::unique_ptr<TransferContext>> create(const Device& device);
    ~D3D12TransferContext() override = default;

    Result<void> stage(const Buffer& dst, std::uint64_t dstOffset, const void* data,
                       std::uint64_t size) override;
    Result<void> flush() override;
    std::uint64_t pendingBytes() const override { return stagingUsed_; }

private:
    struct PendingCopy {
        const Buffer* dst = nullptr;
        std::uint64_t srcOffset = 0;
        std::uint64_t dstOffset = 0;
        std::uint64_t size = 0;
    };

    D3D12TransferContext() = default;
    Result<void> ensureStagingCapacity(std::uint64_t required);

    const D3D12Device* device_ = nullptr;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE fenceEvent_ = nullptr;
    std::uint64_t fenceValue_ = 0;
    std::unique_ptr<Buffer> staging_;
    std::uint64_t stagingUsed_ = 0;
    std::vector<PendingCopy> pending_;
};

// Texture uploads on the direct queue. D3D12 has no blit, so the RGBA8
// mip chain is generated on the CPU (2x2 box filter, in linear space for
// sRGB content — what the Vulkan blit chain computes) and every level is
// copied with its own placed footprint; BC7 payloads copy their stored
// mips. Synchronous per texture, like the Vulkan uploader.
class D3D12TextureUploader final : public TextureUploader {
public:
    static Result<std::unique_ptr<TextureUploader>> create(const Device& device);
    ~D3D12TextureUploader() override;

    Result<std::unique_ptr<Image>> upload(std::uint32_t width, std::uint32_t height,
                                          const void* rgba8, bool srgb) override;
    Result<std::unique_ptr<Image>> uploadCompressed(Format format, const CompressedMip* mips,
                                                    std::uint32_t mipCount, const void* bytes,
                                                    std::uint64_t byteSize) override;

private:
    struct MipSource {
        const std::byte* pixels = nullptr; // tightly packed rows
        std::uint32_t rowBytes = 0;
        std::uint32_t rows = 0; // texel rows (block rows for BC7)
    };

    D3D12TextureUploader() = default;
    Result<void> ensureStagingCapacity(std::uint64_t required);
    Result<std::unique_ptr<Image>> uploadMips(const ImageDesc& desc, std::span<const MipSource> mips);

    const D3D12Device* device_ = nullptr;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE fenceEvent_ = nullptr;
    std::uint64_t fenceValue_ = 0;
    ComPtr<ID3D12Resource> staging_;
    std::uint64_t stagingBytes_ = 0;
    void* stagingMapped_ = nullptr;
};

// The D3D12 frame loop: one command allocator per frame slot, one command
// list re-recorded every frame (static recordings are not implemented
// yet — staticValid_ never becomes true, so static mode just re-records),
// one fence with a value per slot, Present through the swapchain.
class D3D12FrameRenderer final : public FrameRenderer {
public:
    static Result<std::unique_ptr<FrameRenderer>> create(const Device& device,
                                                         Swapchain& swapchain);
    ~D3D12FrameRenderer() override;

    Result<void> drawFrame(const DrawBatch* batch) override;
    Result<void> waitFrameSlot() override;
    void waitIdle() const override;

private:
    D3D12FrameRenderer() = default;

    Result<void> waitForValue(std::uint64_t value, const char* what) const;
    void invalidateStatic() override { staticValid_ = false; }
    Result<void> onSwapchainRecreated(std::uint32_t previousImageCount) override;

    const D3D12Device* dxDevice_ = nullptr;
    D3D12Swapchain* dxSwapchain_ = nullptr;
    std::array<ComPtr<ID3D12CommandAllocator>, kFramesInFlight> allocators_{};
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12Fence> fence_;
    std::array<std::uint64_t, kFramesInFlight> slotValues_{};
    std::uint64_t lastSignalled_ = 0;
    HANDLE fenceEvent_ = nullptr;
};

// Downcasts for the backend's own code: every object created through a
// D3D12 device IS one of these.
inline const D3D12Instance& dx(const Instance& i) { return static_cast<const D3D12Instance&>(i); }
inline const D3D12Device& dx(const Device& d) { return static_cast<const D3D12Device&>(d); }
inline const D3D12Swapchain& dx(const Swapchain& s) { return static_cast<const D3D12Swapchain&>(s); }
inline D3D12Swapchain& dx(Swapchain& s) { return static_cast<D3D12Swapchain&>(s); }
inline const D3D12Buffer& dx(const Buffer& b) { return static_cast<const D3D12Buffer&>(b); }
inline const D3D12Image& dx(const Image& i) { return static_cast<const D3D12Image&>(i); }
inline const D3D12Shader& dx(const Shader& s) { return static_cast<const D3D12Shader&>(s); }
inline const D3D12Pipeline& dx(const Pipeline& p) { return static_cast<const D3D12Pipeline&>(p); }
inline const D3D12DescriptorTable& dx(const DescriptorTable& t) {
    return static_cast<const D3D12DescriptorTable&>(t);
}

// HRESULT -> Error with the failing call's name.
inline Error hrError(const char* what, HRESULT hr) {
    char text[160];
    _snprintf_s(text, sizeof(text), _TRUNCATE, "%s failed (0x%08lx)", what,
                static_cast<unsigned long>(hr));
    return Error{text};
}

} // namespace rend::gpu

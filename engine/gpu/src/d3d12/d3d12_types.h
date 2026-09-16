#pragma once

// The D3D12 backend: concrete classes behind the neutral rend::gpu
// interfaces. Private to engine/gpu — nothing outside the layer includes
// this header, so D3D12/DXGI types may appear freely here.
//
// Slice 2 (skeleton): instance, device on the display adapter, buffers,
// images, a command context that can barrier and clear, a swapchain that
// is a DirectComposition surface when transparent (premultiplied alpha,
// the recipe proven by scratch/tools/d3d11_transparent_test), and the
// frame loop. Everything the scene path needs (shaders, pipelines, the
// descriptor table, uploads, acceleration structures) is slice 3+ and
// still returns "not implemented" from the dispatchers.

#include "rend/gpu/buffer.h"
#include "rend/gpu/command_context.h"
#include "rend/gpu/device.h"
#include "rend/gpu/frame_renderer.h"
#include "rend/gpu/image.h"
#include "rend/gpu/instance.h"
#include "rend/gpu/memory_tracker.h"
#include "rend/gpu/swapchain.h"

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
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rend::gpu {

using Microsoft::WRL::ComPtr;

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
};

class D3D12Buffer final : public Buffer {
public:
    static Result<std::unique_ptr<Buffer>> create(const Device& device, const BufferDesc& desc);
    ~D3D12Buffer() override;

    ID3D12Resource* resource() const { return resource_.Get(); }
    D3D12_GPU_VIRTUAL_ADDRESS gpuAddress() const { return resource_->GetGPUVirtualAddress(); }

private:
    D3D12Buffer() = default;

    ComPtr<ID3D12Resource> resource_;
    std::uint64_t allocatedBytes_ = 0;
    MemoryTracker::Kind trackKind_ = MemoryTracker::Kind::DeviceBuffer;
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
    mutable D3D12_RESOURCE_STATES state_ = D3D12_RESOURCE_STATE_COMMON;
    std::uint64_t allocatedBytes_ = 0;
    bool owned_ = true;
};

// Records into one graphics command list. Skeleton scope: barriers (image
// transitions from the tracked state; global memory barriers become UAV
// barriers), render-target setup + clears, viewport/scissor. Draw, bind
// and dispatch verbs log once and record nothing until slice 3.
class D3D12CommandContext final : public CommandContext {
public:
    explicit D3D12CommandContext(ID3D12GraphicsCommandList* list) : list_(list) {}

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

    ID3D12GraphicsCommandList* list_ = nullptr;
};

// The presentable surface. Transparent targets get a composition
// swapchain (BGRA8, FLIP_SEQUENTIAL, DXGI_ALPHA_MODE_PREMULTIPLIED) shown
// through a DirectComposition visual on the window — the only D3D path
// that composites per-pixel alpha over the desktop. Opaque targets get a
// plain HWND flip swapchain. SwapchainDesc::nativeSurface is the HWND.
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

} // namespace rend::gpu

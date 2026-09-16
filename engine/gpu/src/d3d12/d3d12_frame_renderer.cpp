// D3D12 backend: the frame loop. Per-slot command allocators, one command
// list re-recorded every frame through the neutral recordFrame /
// recordOverlayFrame, a fence value per slot, Present.
#include "d3d12_types.h"

#include "rend/core/log.h"
#include "rend/core/profile.h"

#include <chrono>
#include <format>

namespace rend::gpu {

namespace {

constexpr DWORD kWaitTimeoutMs = 2000;
constexpr int kMaxStalledWaits = 5;

} // namespace

Result<std::unique_ptr<FrameRenderer>> D3D12FrameRenderer::create(const Device& deviceBase,
                                                                  Swapchain& swapchainBase) {
    const D3D12Device& device = dx(deviceBase);
    auto renderer = std::unique_ptr<D3D12FrameRenderer>(new D3D12FrameRenderer());
    renderer->device_ = &device;
    renderer->swapchain_ = &swapchainBase;
    renderer->dxDevice_ = &device;
    renderer->dxSwapchain_ = &dx(swapchainBase);

    for (std::uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (HRESULT hr = device.handle()->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&renderer->allocators_[i]));
            FAILED(hr)) {
            return Error{std::format("CreateCommandAllocator failed (0x{:08x})",
                                     static_cast<unsigned>(hr))};
        }
    }
    if (HRESULT hr = device.handle()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                        renderer->allocators_[0].Get(), nullptr,
                                                        IID_PPV_ARGS(&renderer->list_));
        FAILED(hr)) {
        return Error{std::format("CreateCommandList failed (0x{:08x})", static_cast<unsigned>(hr))};
    }
    renderer->list_->Close(); // created open; drawFrame resets it
    if (HRESULT hr = device.handle()->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                                  IID_PPV_ARGS(&renderer->fence_));
        FAILED(hr)) {
        return Error{std::format("CreateFence failed (0x{:08x})", static_cast<unsigned>(hr))};
    }
    renderer->fenceEvent_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!renderer->fenceEvent_) {
        return Error{"CreateEvent failed"};
    }

    if (auto r = renderer->createDepthBuffer(); !r) {
        return r.error();
    }

    log::info("Frame renderer ready ({} frames in flight, D3D12 direct queue, re-recorded frames)",
              kFramesInFlight);
    return std::unique_ptr<FrameRenderer>(std::move(renderer));
}

Result<void> D3D12FrameRenderer::waitForValue(std::uint64_t value, const char* what) const {
    REND_PROFILE_ZONE("WaitFence");
    if (value == 0 || fence_->GetCompletedValue() >= value) {
        return {};
    }
    if (HRESULT hr = fence_->SetEventOnCompletion(value, fenceEvent_); FAILED(hr)) {
        return Error{std::format("SetEventOnCompletion failed (0x{:08x})", static_cast<unsigned>(hr))};
    }
    for (int attempt = 0;; ++attempt) {
        if (WaitForSingleObject(fenceEvent_, kWaitTimeoutMs) == WAIT_OBJECT_0) {
            return {};
        }
        if (attempt + 1 >= kMaxStalledWaits) {
            return Error{std::format("{} fence never signalled; the GPU appears stalled", what)};
        }
        log::warn("{} still pending after {} s", what, (attempt + 1) * 2);
    }
}

Result<void> D3D12FrameRenderer::waitFrameSlot() {
    return waitForValue(slotValues_[frameIndex_], "Previous frame");
}

void D3D12FrameRenderer::waitIdle() const {
    if (dxDevice_) {
        dxDevice_->waitIdle();
    }
}

Result<void> D3D12FrameRenderer::onSwapchainRecreated(std::uint32_t /*previousImageCount*/) {
    // Fence values are per slot, not per image: nothing to redo.
    return {};
}

Result<void> D3D12FrameRenderer::drawFrame(const DrawBatch* batch) {
    REND_PROFILE_ZONE("DrawFrame");
    if (resizeRequested_) {
        if (pendingWidth_ == 0 || pendingHeight_ == 0) {
            return {}; // minimized: nothing to present to
        }
        if (auto r = recreateSwapchain(); !r) {
            return r.error();
        }
    }

    const std::uint32_t slot = frameIndex_;
    if (auto r = waitForValue(slotValues_[slot], "Previous frame"); !r) {
        return r.error();
    }
    // Presentation pacing: block until the swapchain has room for another
    // frame (max latency kFramesInFlight) instead of stalling in Present.
    if (HANDLE waitable = dxSwapchain_->frameLatencyWaitable()) {
        REND_PROFILE_ZONE("WaitPresent");
        WaitForSingleObjectEx(waitable, 1000, TRUE);
    }

    const std::uint32_t imageIndex = dxSwapchain_->currentBackBufferIndex();

    const auto recordStart = std::chrono::steady_clock::now();
    ID3D12CommandAllocator* allocator = allocators_[slot].Get();
    if (HRESULT hr = allocator->Reset(); FAILED(hr)) {
        return Error{std::format("CommandAllocator::Reset failed (0x{:08x})",
                                 static_cast<unsigned>(hr))};
    }
    if (HRESULT hr = list_->Reset(allocator, nullptr); FAILED(hr)) {
        return Error{std::format("CommandList::Reset failed (0x{:08x})", static_cast<unsigned>(hr))};
    }
    {
        REND_PROFILE_ZONE("RecordScene");
        // Buffers decayed to COMMON when the previous list finished.
        dxDevice_->resetBufferStates();
        D3D12CommandContext ctx(list_.Get(), *dxDevice_);
        recordFrame(ctx, imageIndex, slot, batch);
        // The overlay tail (UI + present transition) follows in the same
        // list: one submission per frame on this backend.
        recordOverlayFrame(ctx, imageIndex);
    }
    if (HRESULT hr = list_->Close(); FAILED(hr)) {
        return Error{std::format("CommandList::Close failed (0x{:08x})", static_cast<unsigned>(hr))};
    }
    stats_.recordMicros += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                              recordStart)
            .count());
    ++stats_.frames;

    {
        REND_PROFILE_ZONE("QueueSubmit");
        ID3D12CommandList* lists[] = {list_.Get()};
        dxDevice_->graphicsQueue()->ExecuteCommandLists(1, lists);
    }
    {
        REND_PROFILE_ZONE("Present");
        const bool vsync = swapchain_->vsync();
        const UINT flags = (!vsync && dxSwapchain_->tearing()) ? DXGI_PRESENT_ALLOW_TEARING : 0;
        const HRESULT presented = dxSwapchain_->handle()->Present(vsync ? 1 : 0, flags);
        if (presented == DXGI_ERROR_DEVICE_REMOVED || presented == DXGI_ERROR_DEVICE_RESET) {
            return Error{std::format("Present: device removed (0x{:08x})",
                                     static_cast<unsigned>(dxDevice_->handle()->GetDeviceRemovedReason()))};
        }
        if (FAILED(presented)) {
            return Error{std::format("Present failed (0x{:08x})", static_cast<unsigned>(presented))};
        }
    }

    const std::uint64_t value = ++lastSignalled_;
    dxDevice_->graphicsQueue()->Signal(fence_.Get(), value);
    slotValues_[slot] = value;

    dxDevice_->drainDebugMessages();
    frameIndex_ = (frameIndex_ + 1) % kFramesInFlight;
    return {};
}

D3D12FrameRenderer::~D3D12FrameRenderer() {
    if (dxDevice_) {
        dxDevice_->waitIdle();
        log::info("Frame renderer destroyed");
    }
    if (fenceEvent_) {
        CloseHandle(fenceEvent_);
    }
}

} // namespace rend::gpu

// D3D12 backend: the presentable surface. Transparent = DirectComposition
// swapchain with premultiplied alpha; opaque = HWND flip swapchain.
#include "d3d12_types.h"

#include "rend/core/log.h"

#include <format>

namespace rend::gpu {

namespace {

// Composition swapchains reject sRGB formats, so both paths present
// through an UNORM BGRA8 surface viewed through sRGB render-target views
// (the documented flip-model trick): pipelines declare the sRGB format
// and the hardware encodes on write.
constexpr DXGI_FORMAT kSurfaceFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
constexpr DXGI_FORMAT kViewFormat = DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
constexpr Format kImageFormat = Format::B8G8R8A8Srgb;
constexpr std::uint32_t kCompositionBuffers = 2;
constexpr std::uint32_t kHwndBuffers = 3;

} // namespace

Result<std::unique_ptr<Swapchain>> D3D12Swapchain::create(const Instance& instanceBase,
                                                          const Device& deviceBase,
                                                          const SwapchainDesc& desc) {
    const D3D12Instance& instance = dx(instanceBase);
    const D3D12Device& device = dx(deviceBase);
    HWND hwnd = static_cast<HWND>(desc.nativeSurface);
    if (!hwnd || !IsWindow(hwnd)) {
        return Error{"D3D12 swapchain needs a window handle in SwapchainDesc::nativeSurface"};
    }

    auto swapchain = std::unique_ptr<D3D12Swapchain>(new D3D12Swapchain());
    swapchain->instance_ = &instance;
    swapchain->device_ = &device;
    swapchain->hwnd_ = hwnd;
    swapchain->transparent_ = desc.transparent;
    swapchain->vsync_ = desc.vsync;
    swapchain->format_ = kImageFormat;

    if (desc.transparent) {
        // The desktop shows through only when the window has NO
        // redirection surface (WS_EX_NOREDIRECTIONBITMAP) and its content
        // comes from the composition visual. The windowing backend created
        // the window (possibly with the Vulkan spike's layered ex-styles),
        // so the style is corrected here, before the composition target.
        LONG_PTR ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
        const LONG_PTR wanted = (ex & ~static_cast<LONG_PTR>(WS_EX_LAYERED)) |
                                WS_EX_NOREDIRECTIONBITMAP;
        if (wanted != ex) {
            SetWindowLongPtrW(hwnd, GWL_EXSTYLE, wanted);
            // Ex-style changes take effect on the next frame change.
            SetWindowPos(hwnd, nullptr, 0, 0, 0, 0,
                         SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE |
                             SWP_FRAMECHANGED);
            log::info("Transparent D3D12 target: ex-style 0x{:x} -> 0x{:x}",
                      static_cast<unsigned long>(ex), static_cast<unsigned long>(wanted));
        }
    }

    if (auto r = swapchain->build(desc.width, desc.height); !r) {
        return r.error();
    }
    return std::unique_ptr<Swapchain>(std::move(swapchain));
}

Result<void> D3D12Swapchain::build(std::uint32_t width, std::uint32_t height) {
    width_ = width;
    height_ = height;
    bufferCount_ = transparent_ ? kCompositionBuffers : kHwndBuffers;

    DXGI_SWAP_CHAIN_DESC1 desc{};
    desc.Width = width;
    desc.Height = height;
    desc.Format = kSurfaceFormat;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = bufferCount_;
    desc.Scaling = DXGI_SCALING_STRETCH;
    desc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    IDXGIFactory6* factory = instance_->factory();

    ComPtr<IDXGISwapChain1> created;
    if (transparent_) {
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        desc.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED;
        if (HRESULT hr = factory->CreateSwapChainForComposition(device_->graphicsQueue(), &desc,
                                                                nullptr, &created);
            FAILED(hr)) {
            return Error{std::format("CreateSwapChainForComposition failed (0x{:08x})",
                                     static_cast<unsigned>(hr))};
        }
    } else {
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        BOOL allowTearing = FALSE;
        ComPtr<IDXGIFactory5> factory5;
        if (SUCCEEDED(factory->QueryInterface(IID_PPV_ARGS(&factory5)))) {
            factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing,
                                          sizeof(allowTearing));
        }
        tearing_ = allowTearing == TRUE;
        if (tearing_) {
            desc.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
        }
        if (HRESULT hr = factory->CreateSwapChainForHwnd(device_->graphicsQueue(), hwnd_, &desc,
                                                         nullptr, nullptr, &created);
            FAILED(hr)) {
            return Error{std::format("CreateSwapChainForHwnd failed (0x{:08x})",
                                     static_cast<unsigned>(hr))};
        }
        factory->MakeWindowAssociation(hwnd_, DXGI_MWA_NO_ALT_ENTER);
    }
    if (HRESULT hr = created.As(&swapchain_); FAILED(hr)) {
        return Error{std::format("IDXGISwapChain3 unavailable (0x{:08x})", static_cast<unsigned>(hr))};
    }
    swapchain_->SetMaximumFrameLatency(FrameRenderer::kFramesInFlight);
    if (waitable_) {
        CloseHandle(waitable_);
    }
    waitable_ = swapchain_->GetFrameLatencyWaitableObject();

    if (transparent_) {
        // DirectComposition binds the swapchain to the window as a visual
        // (DCompositionCreateDevice with no rendering device is enough for
        // a swapchain-only tree).
        if (HRESULT hr = DCompositionCreateDevice(nullptr, IID_PPV_ARGS(&compositionDevice_));
            FAILED(hr)) {
            return Error{std::format("DCompositionCreateDevice failed (0x{:08x})",
                                     static_cast<unsigned>(hr))};
        }
        if (HRESULT hr = compositionDevice_->CreateTargetForHwnd(hwnd_, TRUE, &compositionTarget_);
            FAILED(hr)) {
            return Error{std::format("CreateTargetForHwnd failed (0x{:08x})",
                                     static_cast<unsigned>(hr))};
        }
        if (HRESULT hr = compositionDevice_->CreateVisual(&compositionVisual_); FAILED(hr)) {
            return Error{std::format("CreateVisual failed (0x{:08x})", static_cast<unsigned>(hr))};
        }
        compositionVisual_->SetContent(swapchain_.Get());
        compositionTarget_->SetRoot(compositionVisual_.Get());
        if (HRESULT hr = compositionDevice_->Commit(); FAILED(hr)) {
            return Error{std::format("DirectComposition Commit failed (0x{:08x})",
                                     static_cast<unsigned>(hr))};
        }
    }

    D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heapDesc.NumDescriptors = bufferCount_;
    if (HRESULT hr = device_->handle()->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&rtvHeap_));
        FAILED(hr)) {
        return Error{std::format("CreateDescriptorHeap (swapchain RTVs) failed (0x{:08x})",
                                 static_cast<unsigned>(hr))};
    }

    if (auto r = acquireBuffers(); !r) {
        return r.error();
    }
    log::info("Swapchain {}x{}: {} images, format {}, {}, vsync {}{}", width_, height_, bufferCount_,
              formatName(format_),
              transparent_ ? "composition (premultiplied alpha)" : "hwnd flip", vsync_ ? "on" : "off",
              tearing_ ? ", tearing allowed" : "");
    return {};
}

Result<void> D3D12Swapchain::acquireBuffers() {
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
    for (std::uint32_t i = 0; i < bufferCount_; ++i) {
        ComPtr<ID3D12Resource> buffer;
        if (HRESULT hr = swapchain_->GetBuffer(i, IID_PPV_ARGS(&buffer)); FAILED(hr)) {
            return Error{std::format("GetBuffer({}) failed (0x{:08x})", i, static_cast<unsigned>(hr))};
        }
        D3D12_RENDER_TARGET_VIEW_DESC rtvDesc{};
        rtvDesc.Format = kViewFormat;
        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        device_->handle()->CreateRenderTargetView(buffer.Get(), &rtvDesc, rtv);
        wrapped_.push_back(D3D12Image::wrapExternal(buffer.Get(), rtv, format_, width_, height_));
        buffers_.push_back(std::move(buffer));
        rtv.ptr += device_->rtvDescriptorSize();
    }
    return {};
}

void D3D12Swapchain::releaseBuffers() {
    // Every reference to the buffers must go before ResizeBuffers.
    wrapped_.clear();
    buffers_.clear();
}

Result<void> D3D12Swapchain::recreate(std::uint32_t width, std::uint32_t height) {
    device_->waitIdle();
    releaseBuffers();
    const UINT flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT |
                       (tearing_ ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
    if (HRESULT hr = swapchain_->ResizeBuffers(bufferCount_, width, height, DXGI_FORMAT_UNKNOWN, flags);
        FAILED(hr)) {
        return Error{std::format("ResizeBuffers failed (0x{:08x})", static_cast<unsigned>(hr))};
    }
    width_ = width;
    height_ = height;
    if (auto r = acquireBuffers(); !r) {
        return r.error();
    }
    log::info("Swapchain recreated {}x{} ({} images)", width_, height_, bufferCount_);
    return {};
}

D3D12Swapchain::~D3D12Swapchain() {
    if (device_) {
        device_->waitIdle();
    }
    releaseBuffers();
    compositionVisual_.Reset();
    compositionTarget_.Reset();
    compositionDevice_.Reset();
    if (waitable_) {
        CloseHandle(waitable_);
        waitable_ = nullptr;
    }
    if (swapchain_) {
        log::info("Swapchain destroyed");
    }
}

} // namespace rend::gpu

// D3D12 backend: buffers and images (committed resources).
#include "d3d12_types.h"

#include "rend/core/log.h"

#include <format>

namespace rend::gpu {

// ------------------------------------------------------------------ Buffer

Result<std::unique_ptr<Buffer>> D3D12Buffer::create(const Device& deviceBase,
                                                    const BufferDesc& desc) {
    const D3D12Device& device = dx(deviceBase);
    if (desc.size == 0) {
        return Error{"Buffer size must be non-zero"};
    }

    const bool host = desc.location == MemoryLocation::HostVisible;
    const bool uav = (desc.usage & kUsageStorage) != 0;
    // Buffers the GPU writes (storage, fills) or fetches indirect
    // arguments from need resource states an UPLOAD heap cannot leave
    // GENERIC_READ for; host-visible ones of those live in the device's
    // CPU-visible custom heap instead. Plain staging stays UPLOAD.
    const bool gpuStates =
        (desc.usage & (kUsageStorage | kUsageIndirect | kUsageTransferDst)) != 0;

    D3D12_HEAP_PROPERTIES heap{};
    if (!host) {
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    } else if (gpuStates) {
        heap = device.hostHeapProperties();
    } else {
        heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    }

    D3D12_RESOURCE_DESC resource{};
    resource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resource.Width = desc.size;
    resource.Height = 1;
    resource.DepthOrArraySize = 1;
    resource.MipLevels = 1;
    resource.Format = DXGI_FORMAT_UNKNOWN;
    resource.SampleDesc.Count = 1;
    resource.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // Every storage buffer is a UAV on this backend (shaders declare them
    // RW in all stages — see assets/shaders/backend.hlsli).
    if (uav) {
        resource.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    // Upload-heap resources must start in GENERIC_READ; everything else
    // starts in COMMON and is transitioned explicitly by the recorder.
    const D3D12_RESOURCE_STATES initial =
        heap.Type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ
                                            : D3D12_RESOURCE_STATE_COMMON;

    auto buffer = std::unique_ptr<D3D12Buffer>(new D3D12Buffer());
    if (HRESULT hr = device.handle()->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &resource,
                                                              initial, nullptr,
                                                              IID_PPV_ARGS(&buffer->resource_));
        FAILED(hr)) {
        return Error{std::format("CreateCommittedResource (buffer, {} bytes) failed (0x{:08x})",
                                 desc.size, static_cast<unsigned>(hr))};
    }
    buffer->state_ = initial;

    if (host) {
        const D3D12_RANGE noRead{0, 0};
        if (HRESULT hr = buffer->resource_->Map(0, &noRead, &buffer->mapped_); FAILED(hr)) {
            return Error{std::format("Map failed (0x{:08x})", static_cast<unsigned>(hr))};
        }
    }

    const D3D12_RESOURCE_ALLOCATION_INFO info =
        device.handle()->GetResourceAllocationInfo(0, 1, &resource);
    buffer->device_ = &device;
    buffer->size_ = desc.size;
    buffer->allocatedBytes_ = info.SizeInBytes;
    buffer->trackKind_ = host ? MemoryTracker::Kind::HostBuffer : MemoryTracker::Kind::DeviceBuffer;
    MemoryTracker::onAlloc(buffer->trackKind_, buffer->allocatedBytes_);
    if (gpuStates) {
        buffer->tracked_ = true;
        device.registerBuffer(buffer.get());
    }
    return std::unique_ptr<Buffer>(std::move(buffer));
}

D3D12Buffer::~D3D12Buffer() {
    if (tracked_ && device_) {
        device_->unregisterBuffer(this);
    }
    if (resource_) {
        if (mapped_) {
            resource_->Unmap(0, nullptr);
        }
        MemoryTracker::onFree(trackKind_, allocatedBytes_);
    }
}

// ------------------------------------------------------------------- Image

Result<std::unique_ptr<Image>> D3D12Image::create(const Device& deviceBase, const ImageDesc& desc) {
    const D3D12Device& device = dx(deviceBase);
    if (desc.width == 0 || desc.height == 0) {
        return Error{"Image extent must be non-zero"};
    }
    if (desc.cube) {
        return Error{"D3D12 backend: cube images not implemented yet"};
    }
    const DXGI_FORMAT format = toDxgi(desc.format);
    if (format == DXGI_FORMAT_UNKNOWN) {
        return Error{std::format("D3D12 backend: unsupported image format {}", formatName(desc.format))};
    }

    const bool color = (desc.usage & kImageUsageColorAttachment) != 0;
    const bool depth = desc.depth || (desc.usage & kImageUsageDepthAttachment) != 0;
    const bool sampled = (desc.usage & kImageUsageSampled) != 0;

    // A depth image the shaders sample (cascade maps) is a typeless
    // resource viewed as D32_FLOAT by the DSV and R32_FLOAT by the SRV.
    DXGI_FORMAT resourceFormat = format;
    DXGI_FORMAT srvFormat = format;
    if (depth && sampled) {
        resourceFormat = DXGI_FORMAT_R32_TYPELESS;
        srvFormat = DXGI_FORMAT_R32_FLOAT;
    }

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC resource{};
    resource.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resource.Width = desc.width;
    resource.Height = desc.height;
    resource.DepthOrArraySize = 1;
    resource.MipLevels = static_cast<UINT16>(desc.mipLevels);
    resource.Format = resourceFormat;
    resource.SampleDesc.Count = 1;
    resource.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    if (color) {
        resource.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }
    if (depth) {
        resource.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        if (!sampled) {
            resource.Flags |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
        }
    }

    // Every depth clear in the engine is 1.0. Colour targets get no
    // optimized clear value: their clears differ per pass (zero G-buffer,
    // the scene clear colour), and a mismatching value only costs a
    // debug-layer warning per clear.
    D3D12_CLEAR_VALUE clear{};
    clear.Format = format;
    clear.DepthStencil.Depth = 1.0f;
    const D3D12_CLEAR_VALUE* clearPtr = depth ? &clear : nullptr;

    auto out = std::unique_ptr<D3D12Image>(new D3D12Image());
    if (HRESULT hr = device.handle()->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &resource, D3D12_RESOURCE_STATE_COMMON, clearPtr,
            IID_PPV_ARGS(&out->resource_));
        FAILED(hr)) {
        return Error{std::format("CreateCommittedResource (image {}x{} {}) failed (0x{:08x})",
                                 desc.width, desc.height, formatName(desc.format),
                                 static_cast<unsigned>(hr))};
    }

    if (color) {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        heapDesc.NumDescriptors = 1;
        if (HRESULT hr = device.handle()->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&out->rtvHeap_));
            FAILED(hr)) {
            return Error{std::format("CreateDescriptorHeap (RTV) failed (0x{:08x})",
                                     static_cast<unsigned>(hr))};
        }
        out->rtv_ = out->rtvHeap_->GetCPUDescriptorHandleForHeapStart();
        device.handle()->CreateRenderTargetView(out->resource_.Get(), nullptr, out->rtv_);
    }
    if (depth) {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        heapDesc.NumDescriptors = 1;
        if (HRESULT hr = device.handle()->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&out->dsvHeap_));
            FAILED(hr)) {
            return Error{std::format("CreateDescriptorHeap (DSV) failed (0x{:08x})",
                                     static_cast<unsigned>(hr))};
        }
        out->dsv_ = out->dsvHeap_->GetCPUDescriptorHandleForHeapStart();
        D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
        dsvDesc.Format = format;
        dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        device.handle()->CreateDepthStencilView(out->resource_.Get(), &dsvDesc, out->dsv_);
    }

    const D3D12_RESOURCE_ALLOCATION_INFO info =
        device.handle()->GetResourceAllocationInfo(0, 1, &resource);
    out->format_ = desc.format;
    out->width_ = desc.width;
    out->height_ = desc.height;
    out->mipLevels_ = desc.mipLevels;
    out->layerCount_ = 1;
    out->srvFormat_ = srvFormat;
    out->allocatedBytes_ = info.SizeInBytes;
    MemoryTracker::onAlloc(MemoryTracker::Kind::Image, out->allocatedBytes_);
    return std::unique_ptr<Image>(std::move(out));
}

std::unique_ptr<Image> D3D12Image::wrapExternal(ID3D12Resource* resource,
                                                D3D12_CPU_DESCRIPTOR_HANDLE rtv, Format format,
                                                std::uint32_t width, std::uint32_t height) {
    auto out = std::unique_ptr<D3D12Image>(new D3D12Image());
    out->resource_ = resource;
    out->rtv_ = rtv;
    out->format_ = format;
    out->srvFormat_ = toDxgi(format);
    out->width_ = width;
    out->height_ = height;
    out->owned_ = false;
    // Swapchain buffers are handed out in the PRESENT (COMMON) state and
    // return to it at the end of every frame.
    out->state_ = D3D12_RESOURCE_STATE_PRESENT;
    return out;
}

D3D12_SHADER_RESOURCE_VIEW_DESC D3D12Image::srvDesc() const {
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = srvFormat_;
    desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.Texture2D.MostDetailedMip = 0;
    desc.Texture2D.MipLevels = mipLevels_;
    return desc;
}

D3D12Image::~D3D12Image() {
    if (owned_ && resource_) {
        MemoryTracker::onFree(MemoryTracker::Kind::Image, allocatedBytes_);
    }
}

} // namespace rend::gpu

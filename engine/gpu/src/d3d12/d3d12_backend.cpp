// D3D12 backend: instance (DXGI factory + debug layer) and device
// (adapter selection, queues, feature report, idle fence, debug drain).
#include "d3d12_types.h"

#include "rend/core/log.h"

#include <format>
#include <string>

namespace rend::gpu {

namespace {

std::string narrow(const wchar_t* wide) {
    if (!wide || !*wide) {
        return {};
    }
    const int length = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<std::size_t>(length > 0 ? length - 1 : 0), '\0');
    if (length > 1) {
        WideCharToMultiByte(CP_UTF8, 0, wide, -1, out.data(), length, nullptr, nullptr);
    }
    return out;
}

// What D3D12 (feature level 12.0 + the queried options) offers for each
// engine feature. Several Vulkan-specific notions have no D3D12 analogue
// and are reported as present because the equivalent behaviour is
// always available (dynamic rendering, modern barriers, demote) or as
// absent because nothing maps (buffer device address).
bool supports(ID3D12Device* device, Feature f) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options));
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5{};
    device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5));
    switch (f) {
    case Feature::MultiDrawIndirect:
    case Feature::DrawIndirectFirstInstance:
    case Feature::DrawIndirectCount:
    case Feature::ShaderDrawParameters:
        return true; // ExecuteIndirect with a count buffer + SV_ instance ids
    case Feature::DescriptorIndexing:
        return options.ResourceBindingTier >= D3D12_RESOURCE_BINDING_TIER_3;
    case Feature::TimelineSemaphore:
    case Feature::DynamicRendering:
    case Feature::Synchronization2:
    case Feature::ShaderDemote:
    case Feature::FragmentStores:
        return true;
    case Feature::BufferDeviceAddress:
        return false;
    case Feature::AccelerationStructure:
    case Feature::RayQuery:
        return options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1;
    case Feature::Count:
        break;
    }
    return false;
}

} // namespace

DXGI_FORMAT toDxgi(Format format) {
    switch (format) {
    case Format::R8G8B8A8Unorm: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case Format::R8G8B8A8Srgb: return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case Format::B8G8R8A8Unorm: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case Format::B8G8R8A8Srgb: return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    case Format::R16G16B16A16Sfloat: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case Format::R32Sfloat: return DXGI_FORMAT_R32_FLOAT;
    case Format::R32G32Sfloat: return DXGI_FORMAT_R32G32_FLOAT;
    case Format::R32G32B32Sfloat: return DXGI_FORMAT_R32G32B32_FLOAT;
    case Format::D32Sfloat: return DXGI_FORMAT_D32_FLOAT;
    case Format::Bc7Unorm: return DXGI_FORMAT_BC7_UNORM;
    case Format::Bc7Srgb: return DXGI_FORMAT_BC7_UNORM_SRGB;
    case Format::Undefined: break;
    }
    return DXGI_FORMAT_UNKNOWN;
}

// ---------------------------------------------------------------- Instance

Result<std::unique_ptr<Instance>> D3D12Instance::create(const InstanceDesc& desc) {
    auto instance = std::unique_ptr<D3D12Instance>(new D3D12Instance());

    UINT factoryFlags = 0;
    if (desc.enableValidation) {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            instance->debug_ = true;
            factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
        } else {
            log::warn("D3D12 debug layer requested but unavailable (install Graphics Tools); "
                      "continuing without it");
        }
    }

    if (HRESULT hr = CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&instance->factory_)); FAILED(hr)) {
        return Error{std::format("CreateDXGIFactory2 failed (0x{:08x})", static_cast<unsigned>(hr))};
    }

    log::info("D3D12 instance created (DXGI factory, debug layer {}, sync validation {})",
              instance->debug_ ? "on" : "off",
              desc.enableSyncValidation ? "n/a (GPU-based validation not enabled)" : "off");
    return std::unique_ptr<Instance>(std::move(instance));
}

// ------------------------------------------------------------------ Device

Result<std::unique_ptr<Device>> D3D12Device::create(const Instance& instanceBase,
                                                    const FeatureSet& request) {
    const D3D12Instance& instance = dx(instanceBase);
    IDXGIFactory6* factory = instance.factory();

    // Adapter scoring mirrors the Vulkan backend: the adapter that owns a
    // display wins outright (presenting from a display-less adapter takes
    // the cross-adapter copy path that wedges some AMD iGPU+dGPU boxes),
    // then hardware over software, then dedicated memory.
    ComPtr<IDXGIAdapter1> best;
    DXGI_ADAPTER_DESC1 bestDesc{};
    long long bestScore = -1;
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i, adapter.Reset()) {
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device),
                                     nullptr))) {
            log::trace("{}: no feature level 12.0 device", narrow(desc.Description));
            continue;
        }
        ComPtr<IDXGIOutput> output;
        const bool hasOutput = adapter->EnumOutputs(0, &output) == S_OK;
        long long score = 1000;
        if (hasOutput) {
            score += 10000;
        }
        score += static_cast<long long>(desc.DedicatedVideoMemory >> 30); // GiB tie-break
        log::info("Adapter '{}': score {} (display owner {})", narrow(desc.Description), score,
                  hasOutput ? "yes" : "no");
        if (score > bestScore) {
            bestScore = score;
            best = adapter;
            bestDesc = desc;
        }
    }
    if (!best) {
        return Error{"No D3D12 feature level 12.0 hardware adapter found"};
    }

    auto device = std::unique_ptr<D3D12Device>(new D3D12Device());
    device->adapter_ = best;
    device->adapterName_ = narrow(bestDesc.Description);
    if (HRESULT hr = D3D12CreateDevice(best.Get(), D3D_FEATURE_LEVEL_12_0,
                                       IID_PPV_ARGS(&device->device_));
        FAILED(hr)) {
        return Error{std::format("D3D12CreateDevice failed (0x{:08x})", static_cast<unsigned>(hr))};
    }

    if (instance.validationEnabled()) {
        if (SUCCEEDED(device->device_.As(&device->infoQueue_))) {
            // Never break into the debugger: messages are drained into the
            // log once per frame instead (drainDebugMessages).
            device->infoQueue_->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, FALSE);
            device->infoQueue_->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, FALSE);
        }
    }

    // Feature report: required ones fail creation, optional ones are
    // enabled when present (there is nothing to "enable" on D3D12 — the
    // mask just records what the adapter can do).
    std::uint64_t mask = 0;
    for (Feature f : request.required) {
        if (!supports(device->device_.Get(), f)) {
            return Error{std::format("{}: missing required feature {}", device->adapterName_,
                                     featureName(f))};
        }
        mask |= 1ull << static_cast<std::uint32_t>(f);
    }
    for (Feature f : request.optional) {
        if (supports(device->device_.Get(), f)) {
            mask |= 1ull << static_cast<std::uint32_t>(f);
        } else {
            log::warn("{}: optional feature {} unavailable", device->adapterName_, featureName(f));
        }
    }
    device->enabledMask_ = mask;
    device->maxSamplerAnisotropy_ = 16.0f;

    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (HRESULT hr = device->device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&device->graphics_));
        FAILED(hr)) {
        return Error{std::format("CreateCommandQueue (direct) failed (0x{:08x})",
                                 static_cast<unsigned>(hr))};
    }
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
    if (HRESULT hr = device->device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&device->copy_));
        FAILED(hr)) {
        return Error{std::format("CreateCommandQueue (copy) failed (0x{:08x})",
                                 static_cast<unsigned>(hr))};
    }
    if (HRESULT hr = device->device_->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                                  IID_PPV_ARGS(&device->idleFence_));
        FAILED(hr)) {
        return Error{std::format("CreateFence failed (0x{:08x})", static_cast<unsigned>(hr))};
    }
    device->idleEvent_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!device->idleEvent_) {
        return Error{"CreateEvent failed"};
    }
    device->rtvSize_ = device->device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    device->dsvSize_ = device->device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

    log::info("Device created on '{}' (D3D12 feature level 12.0, direct + copy queues)",
              device->adapterName_);
    return std::unique_ptr<Device>(std::move(device));
}

void D3D12Device::waitIdle() const {
    if (!idleFence_) {
        return;
    }
    for (ID3D12CommandQueue* queue : {graphics_.Get(), copy_.Get()}) {
        const std::uint64_t value = ++idleValue_;
        queue->Signal(idleFence_.Get(), value);
        if (idleFence_->GetCompletedValue() < value) {
            idleFence_->SetEventOnCompletion(value, idleEvent_);
            WaitForSingleObject(idleEvent_, INFINITE);
        }
    }
}

void D3D12Device::drainDebugMessages() const {
    if (!infoQueue_) {
        return;
    }
    const UINT64 count = infoQueue_->GetNumStoredMessages();
    std::vector<char> storage;
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T length = 0;
        if (FAILED(infoQueue_->GetMessage(i, nullptr, &length)) || length == 0) {
            continue;
        }
        storage.resize(length);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (FAILED(infoQueue_->GetMessage(i, message, &length))) {
            continue;
        }
        const std::string_view text(message->pDescription, message->DescriptionByteLength > 0
                                                               ? message->DescriptionByteLength - 1
                                                               : 0);
        switch (message->Severity) {
        case D3D12_MESSAGE_SEVERITY_CORRUPTION:
        case D3D12_MESSAGE_SEVERITY_ERROR: log::error("[d3d12] {}", text); break;
        case D3D12_MESSAGE_SEVERITY_WARNING: log::warn("[d3d12] {}", text); break;
        default: log::trace("[d3d12] {}", text); break;
        }
    }
    infoQueue_->ClearStoredMessages();
}

D3D12Device::~D3D12Device() {
    if (device_) {
        waitIdle();
        drainDebugMessages();
        log::info("Device destroyed ('{}')", adapterName_);
    }
    if (idleEvent_) {
        CloseHandle(idleEvent_);
    }
}

} // namespace rend::gpu

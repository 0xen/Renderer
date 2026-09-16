// D3D12 backend: the bindless table as descriptor heaps plus the shared
// root signature. Binding N of the Vulkan set 0 is register space N here
// (assets/shaders/backend.hlsli), and the binding list below is the
// D3D12 twin of the layout in descriptor_table.cpp — new bindings go in
// both, with the structured stride the HLSL declares.
#include "d3d12_types.h"

#include "rend/core/log.h"

#include <algorithm>
#include <format>

namespace rend::gpu {

namespace {

using Kind = D3D12DescriptorTable::Kind;
constexpr Kind U = Kind::UavBuffer; // shader-written (or aliasing a written buffer)
constexpr Kind B = Kind::SrvBuffer; // read-only buffer
constexpr Kind I = Kind::SrvImage;  // sampled image / TLAS

struct BindingSpec {
    std::uint32_t binding;
    Kind kind;
    std::uint32_t count;       // array size (0 = the table's maxTextures)
    std::uint32_t stride;      // structured stride; 0 = raw (ByteAddressBuffer)
    bool cube = false;         // TextureCube SRV
    bool depth = false;        // sampled depth (R32_FLOAT over typeless)
    bool rayQueryOnly = false; // present only when Feature::RayQuery is on
};

// Strides are the std430 sizes of the HLSL structs the shaders declare
// (main.cpp static_asserts the C++ twins): ObjectData 48, DrawCommand 24,
// CameraData 128, LightData 1184, uint4 16, SkinVertex 24, JointMatrix 64,
// float4x4 64, InstanceRow 8, ObjectBounds 32, MeshLodTable 80.
constexpr BindingSpec kBindings[] = {
    {0, B, 1, 48},
    {1, I, 0, 0}, // bindless textures[maxTextures]
    {3, B, 1, 24},
    {4, U, 1, 24},
    {5, U, 1, 4},
    {6, B, 1, 128},
    {7, B, 1, 1184},
    {8, I, 4, 0, false, true}, // shadow cascades
    {10, I, 1, 0, false, false, true},  // TLAS
    {11, U, 1, 0, false, false, true},  // pool raw bytes (RT fetch; aliases 13)
    {12, B, 1, 16, false, false, true}, // geometry info
    {13, U, 1, 0},                      // pool raw bytes (skin writes, obb reads)
    {14, B, 1, 24},
    {15, B, 1, 64},
    {16, B, 1, 4},
    {17, B, 1, 4},
    {18, I, 1, 0, true}, // probe cube
    {19, B, 1, 64},
    {20, U, 1, 8}, // instance rows (aliases 23, which the cull pass writes)
    {21, U, 1, 24},
    {22, B, 1, 32},
    {23, U, 1, 8},
    {24, U, 1, 24},
    {25, B, 1, 80},
    {26, U, 1, 4},
    {27, I, 16, 0, true}, // point-light shadow cubes
    {28, I, 1, 0},
    {29, I, 1, 0},
    {30, I, 1, 0},
    {31, I, 1, 0},
    {32, B, 1, 4, false, false, true}, // RT hit remap
    {33, U, 1, 0},                     // refined OBBs, raw (refine writes)
    {34, B, 1, 4},
    {35, I, 1, 0},
    {36, I, 1, 0},
};

void logOnce(const char* what) {
    static const char* seen[8] = {};
    for (const char*& s : seen) {
        if (s == what) {
            return;
        }
        if (s == nullptr) {
            s = what;
            log::error("D3D12 backend: {}", what);
            return;
        }
    }
}

} // namespace

Result<std::unique_ptr<DescriptorTable>> D3D12DescriptorTable::create(
    const Device& deviceBase, const DescriptorTableDesc& desc) {
    const D3D12Device& device = dx(deviceBase);
    ID3D12Device* d3d = device.handle();
    auto table = std::unique_ptr<D3D12DescriptorTable>(new D3D12DescriptorTable());
    table->device_ = &device;
    table->userStorageBuffers_ = desc.userStorageBuffers;
    table->userSampledImages_ = desc.userSampledImages;
    const bool rayQuery = device.isEnabled(Feature::RayQuery);

    // Slot layout: the engine bindings in order, then the caller's user
    // bindings (storage buffers first, then sampled images).
    std::uint32_t next = 0;
    for (const BindingSpec& spec : kBindings) {
        if (spec.rayQueryOnly && !rayQuery) {
            continue;
        }
        Slot slot{};
        slot.binding = spec.binding;
        slot.kind = spec.kind;
        slot.cube = spec.cube;
        slot.depth = spec.depth;
        slot.count = spec.count == 0 ? std::max(desc.maxTextures, 1u) : spec.count;
        slot.stride = spec.stride;
        slot.first = next;
        next += slot.count;
        table->slots_.push_back(slot);
    }
    for (std::uint32_t i = 0; i < desc.userStorageBuffers; ++i) {
        table->slots_.push_back(Slot{.binding = table->userStorageBinding(i),
                                     .kind = Kind::UavBuffer,
                                     .count = 1,
                                     .first = next++});
    }
    for (std::uint32_t i = 0; i < desc.userSampledImages; ++i) {
        table->slots_.push_back(
            Slot{.binding = table->userSampledImageBinding(i), .count = 1, .first = next++});
    }
    const std::uint32_t slotCount = next;

    D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = slotCount;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (HRESULT hr = d3d->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&table->resourceHeap_));
        FAILED(hr)) {
        return hrError("CreateDescriptorHeap (CBV_SRV_UAV)", hr);
    }
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    heapDesc.NumDescriptors = 2;
    if (HRESULT hr = d3d->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&table->samplerHeap_));
        FAILED(hr)) {
        return hrError("CreateDescriptorHeap (sampler)", hr);
    }

    // Every slot starts as a null descriptor of its kind so an unwritten
    // (partially bound) binding reads zeros instead of garbage.
    for (const Slot& slot : table->slots_) {
        for (std::uint32_t i = 0; i < slot.count; ++i) {
            const D3D12_CPU_DESCRIPTOR_HANDLE handle = table->cpuHandle(slot.first + i);
            if (slot.kind == Kind::UavBuffer) {
                D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
                uav.Format = DXGI_FORMAT_R32_TYPELESS;
                uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                uav.Buffer.NumElements = 1;
                uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
                d3d->CreateUnorderedAccessView(nullptr, nullptr, &uav, handle);
            } else if (slot.kind == Kind::SrvBuffer) {
                D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
                srv.Format = DXGI_FORMAT_R32_TYPELESS;
                srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                srv.Buffer.NumElements = 1;
                srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
                d3d->CreateShaderResourceView(nullptr, &srv, handle);
            } else if (slot.binding == 10) {
                D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
                srv.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
                srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                d3d->CreateShaderResourceView(nullptr, &srv, handle);
            } else {
                D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
                srv.Format = slot.depth ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
                srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                if (slot.cube) {
                    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
                    srv.TextureCube.MipLevels = 1;
                } else {
                    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                    srv.Texture2D.MipLevels = 1;
                }
                d3d->CreateShaderResourceView(nullptr, &srv, handle);
            }
        }
    }

    // Samplers: binding 2 = trilinear repeat (anisotropic when offered),
    // binding 9 = the PCF comparison sampler (border white = lit).
    {
        D3D12_SAMPLER_DESC linear{};
        linear.Filter = device.maxSamplerAnisotropy() > 0.0f ? D3D12_FILTER_ANISOTROPIC
                                                              : D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        linear.AddressU = linear.AddressV = linear.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        linear.MaxAnisotropy =
            static_cast<UINT>(std::min(8.0f, std::max(1.0f, device.maxSamplerAnisotropy())));
        linear.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        linear.MaxLOD = D3D12_FLOAT32_MAX;
        D3D12_CPU_DESCRIPTOR_HANDLE handle = table->samplerHeap_->GetCPUDescriptorHandleForHeapStart();
        d3d->CreateSampler(&linear, handle);

        D3D12_SAMPLER_DESC shadow{};
        shadow.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
        shadow.AddressU = shadow.AddressV = shadow.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
        shadow.BorderColor[0] = shadow.BorderColor[1] = shadow.BorderColor[2] =
            shadow.BorderColor[3] = 1.0f;
        shadow.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
        shadow.MaxAnisotropy = 1;
        shadow.MaxLOD = D3D12_FLOAT32_MAX;
        handle.ptr += device.samplerDescriptorSize();
        d3d->CreateSampler(&shadow, handle);
    }

    // Root signature: [0] push block b0, [1] base instance b1, [2] the
    // resource table (one range per binding, offset = its heap slot),
    // [3] the two samplers.
    std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
    ranges.reserve(table->slots_.size());
    for (const Slot& slot : table->slots_) {
        ranges.push_back(D3D12_DESCRIPTOR_RANGE{
            .RangeType = slot.kind == Kind::UavBuffer ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV
                                                      : D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            .NumDescriptors = slot.count,
            .BaseShaderRegister = 0,
            .RegisterSpace = slot.binding,
            .OffsetInDescriptorsFromTableStart = slot.first,
        });
    }
    const D3D12_DESCRIPTOR_RANGE samplerRanges[2] = {
        {D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 2, 0},
        {D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 9, 1},
    };
    D3D12_ROOT_PARAMETER params[4]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, kPushDwords};
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[1].Constants = {1, 0, 1};
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable = {static_cast<UINT>(ranges.size()), ranges.data()};
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[3].DescriptorTable = {2, samplerRanges};
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 4;
    rsDesc.pParameters = params;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> errors;
    if (HRESULT hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors);
        FAILED(hr)) {
        return Error{std::format("D3D12SerializeRootSignature failed (0x{:08x}): {}",
                                 static_cast<unsigned>(hr),
                                 errors ? static_cast<const char*>(errors->GetBufferPointer()) : "")};
    }
    if (HRESULT hr = d3d->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                              IID_PPV_ARGS(&table->rootSignature_));
        FAILED(hr)) {
        return hrError("CreateRootSignature", hr);
    }

    // Indirect draws: each DrawIndexedIndirect record's leading word
    // becomes root parameter 1 (the base instance), then the five draw
    // arguments follow.
    D3D12_INDIRECT_ARGUMENT_DESC args[2]{};
    args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
    args[0].Constant.RootParameterIndex = 1;
    args[0].Constant.DestOffsetIn32BitValues = 0;
    args[0].Constant.Num32BitValuesToSet = 1;
    args[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
    D3D12_COMMAND_SIGNATURE_DESC sigDesc{};
    sigDesc.ByteStride = kIndirectStride;
    sigDesc.NumArgumentDescs = 2;
    sigDesc.pArgumentDescs = args;
    if (HRESULT hr = d3d->CreateCommandSignature(&sigDesc, table->rootSignature_.Get(),
                                                 IID_PPV_ARGS(&table->drawIndexedSignature_));
        FAILED(hr)) {
        return hrError("CreateCommandSignature (draw indexed)", hr);
    }

    log::info("Descriptor table ready (D3D12: {} heap slots, {} bindings, {} textures)", slotCount,
              table->slots_.size(), desc.maxTextures);
    return std::unique_ptr<DescriptorTable>(std::move(table));
}

const D3D12DescriptorTable::Slot* D3D12DescriptorTable::slot(std::uint32_t binding) const {
    for (const Slot& s : slots_) {
        if (s.binding == binding) {
            return &s;
        }
    }
    log::error("D3D12 backend: descriptor binding {} does not exist in this table", binding);
    return nullptr;
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12DescriptorTable::cpuHandle(std::uint32_t index) const {
    D3D12_CPU_DESCRIPTOR_HANDLE handle = resourceHeap_->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(index) * device_->resourceDescriptorSize();
    return handle;
}

void D3D12DescriptorTable::writeBufferView(const Slot& slot, const Buffer& buffer,
                                           std::uint64_t range) {
    if (slot.kind == Kind::SrvImage) {
        log::error("D3D12 backend: binding {} is not a storage buffer", slot.binding);
        return;
    }
    const std::uint64_t bytes = range == 0 ? buffer.size() : range;
    const bool raw = slot.stride == 0;
    const UINT elements = static_cast<UINT>(bytes / (raw ? 4 : slot.stride));
    if (slot.kind == Kind::UavBuffer) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        desc.Format = raw ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
        desc.Buffer.NumElements = elements;
        desc.Buffer.StructureByteStride = raw ? 0 : slot.stride;
        desc.Buffer.Flags = raw ? D3D12_BUFFER_UAV_FLAG_RAW : D3D12_BUFFER_UAV_FLAG_NONE;
        device_->handle()->CreateUnorderedAccessView(dx(buffer).resource(), nullptr, &desc,
                                                     cpuHandle(slot.first));
        // One resource state per buffer: written anywhere = UAV state for
        // every shader access (the recorder reads this flag).
        dx(buffer).markUavBound();
    } else {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
        desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        desc.Format = raw ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.Buffer.NumElements = elements;
        desc.Buffer.StructureByteStride = raw ? 0 : slot.stride;
        desc.Buffer.Flags = raw ? D3D12_BUFFER_SRV_FLAG_RAW : D3D12_BUFFER_SRV_FLAG_NONE;
        device_->handle()->CreateShaderResourceView(dx(buffer).resource(), &desc,
                                                    cpuHandle(slot.first));
    }
}

void D3D12DescriptorTable::writeImageView(const Slot& slot, std::uint32_t index,
                                          const Image& image) {
    if (slot.kind != Kind::SrvImage) {
        log::error("D3D12 backend: binding {} is not a sampled image", slot.binding);
        return;
    }
    if (index >= slot.count) {
        log::error("D3D12 backend: binding {} index {} out of range ({})", slot.binding, index,
                   slot.count);
        return;
    }
    D3D12_SHADER_RESOURCE_VIEW_DESC desc = dx(image).srvDesc();
    if (slot.cube) {
        desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
        desc.TextureCube.MostDetailedMip = 0;
        desc.TextureCube.MipLevels = image.mipLevels();
    }
    device_->handle()->CreateShaderResourceView(dx(image).resource(), &desc,
                                                cpuHandle(slot.first + index));
}

void D3D12DescriptorTable::writeObjectBuffer(const Buffer& buffer, std::uint64_t range) {
    writeStorageBuffer(0, buffer, range);
}

void D3D12DescriptorTable::writeStorageBuffer(std::uint32_t binding, const Buffer& buffer,
                                              std::uint64_t range) {
    if (const Slot* s = slot(binding)) {
        writeBufferView(*s, buffer, range);
    }
}

void D3D12DescriptorTable::writeTexture(std::uint32_t index, const Image& image) {
    writeSampledImage(1, index, image);
}

void D3D12DescriptorTable::writeShadowMap(std::uint32_t cascade, const Image& image) {
    writeSampledImage(8, cascade, image);
}

void D3D12DescriptorTable::writeProbe(const Image& image) { writeSampledImage(18, 0, image); }

void D3D12DescriptorTable::writePointShadowMap(std::uint32_t index, const Image& image) {
    writeSampledImage(27, index, image);
}

void D3D12DescriptorTable::writeSampledImage(std::uint32_t binding, std::uint32_t index,
                                             const Image& image) {
    if (const Slot* s = slot(binding)) {
        writeImageView(*s, index, image);
    }
}

void D3D12DescriptorTable::writeAccelerationStructure(const AccelerationStructure& /*tlas*/) {
    logOnce("acceleration structures are not implemented yet (binding 10 stays null)");
}

} // namespace rend::gpu

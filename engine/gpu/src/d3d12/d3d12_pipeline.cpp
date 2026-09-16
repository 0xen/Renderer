// D3D12 backend: DXIL shaders and pipeline state objects.
#include "d3d12_types.h"

#include "rend/core/log.h"

#include <format>
#include <fstream>

namespace rend::gpu {

// ------------------------------------------------------------------ Shader

Result<std::unique_ptr<Shader>> D3D12Shader::createFromFile(const Device& /*device*/,
                                                            const std::filesystem::path& path) {
    // Callers name the SPIR-V artifact; the DXIL build of the same stage
    // sits beside it (assets/CMakeLists.txt emits both).
    std::filesystem::path dxil = path;
    if (dxil.extension() == ".spv") {
        dxil.replace_extension(".dxil");
    }
    std::ifstream file(dxil, std::ios::binary | std::ios::ate);
    if (!file) {
        return Error{std::format("Shader file not found: {}", dxil.string())};
    }
    const std::streamoff size = file.tellg();
    if (size <= 0) {
        return Error{std::format("Shader file is empty: {}", dxil.string())};
    }
    auto shader = std::unique_ptr<D3D12Shader>(new D3D12Shader());
    shader->code_.resize(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(shader->code_.data()), size);
    if (!file) {
        return Error{std::format("Failed to read shader file: {}", dxil.string())};
    }
    log::trace("Shader loaded: {} ({} bytes)", dxil.filename().string(),
               static_cast<long long>(size));
    return std::unique_ptr<Shader>(std::move(shader));
}

// ---------------------------------------------------------------- Pipeline

namespace {

// Vertex attribute locations -> DXIL semantics. dxc assigns SPIR-V
// locations in declaration order, so the engine's interleaved
// position/normal/uv layout is locations 0/1/2 (scene.hlsl VSInput).
struct Semantic {
    const char* name;
    UINT index;
};

Semantic semanticOf(std::uint32_t location) {
    switch (location) {
    case 0: return {"POSITION", 0};
    case 1: return {"NORMAL", 0};
    case 2: return {"TEXCOORD", 0};
    default: return {"TEXCOORD", location - 2};
    }
}

} // namespace

Result<std::unique_ptr<Pipeline>> D3D12Pipeline::createGraphics(const Device& deviceBase,
                                                                const GraphicsPipelineDesc& desc) {
    const D3D12Device& device = dx(deviceBase);
    if (!desc.vertexShader || !desc.fragmentShader) {
        return Error{"Graphics pipeline needs a vertex and a fragment shader"};
    }
    if (!desc.descriptorTable) {
        return Error{"D3D12 backend: a graphics pipeline needs the descriptor table (it owns the "
                     "root signature)"};
    }
    const D3D12DescriptorTable& table = dx(*desc.descriptorTable);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = table.rootSignature();
    pso.VS = dx(*desc.vertexShader).bytecode();
    pso.PS = dx(*desc.fragmentShader).bytecode();

    std::vector<D3D12_INPUT_ELEMENT_DESC> elements;
    if (desc.vertexStride > 0) {
        elements.reserve(desc.vertexAttributes.size());
        for (const VertexAttribute& a : desc.vertexAttributes) {
            const Semantic semantic = semanticOf(a.location);
            elements.push_back(D3D12_INPUT_ELEMENT_DESC{
                .SemanticName = semantic.name,
                .SemanticIndex = semantic.index,
                .Format = toDxgi(a.format),
                .InputSlot = 0,
                .AlignedByteOffset = a.offset,
                .InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                .InstanceDataStepRate = 0,
            });
        }
    }
    pso.InputLayout = {elements.data(), static_cast<UINT>(elements.size())};

    // Blend / write-mask state is identical across attachments (MRT
    // replicates attachment 0 with IndependentBlendEnable off).
    D3D12_RENDER_TARGET_BLEND_DESC blend{};
    blend.BlendEnable = (desc.alphaBlend || desc.occlusionDebug) ? TRUE : FALSE;
    blend.SrcBlend = D3D12_BLEND_SRC_ALPHA;
    blend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    blend.BlendOp = D3D12_BLEND_OP_ADD;
    blend.SrcBlendAlpha = D3D12_BLEND_ONE;
    blend.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
    blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    blend.LogicOp = D3D12_LOGIC_OP_NOOP;
    blend.RenderTargetWriteMask = desc.occlusionProxy ? 0 : D3D12_COLOR_WRITE_ENABLE_ALL;
    pso.BlendState.AlphaToCoverageEnable = FALSE;
    pso.BlendState.IndependentBlendEnable = FALSE;
    pso.BlendState.RenderTarget[0] = blend;
    pso.SampleMask = UINT_MAX;

    // Cull NONE everywhere: the projections bake a Y flip that reverses
    // winding (and rendClip flips it back for DXIL).
    pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso.RasterizerState.FrontCounterClockwise = FALSE;
    pso.RasterizerState.DepthClipEnable = TRUE;
    pso.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

    if (desc.depthFormat != Format::Undefined) {
        pso.DepthStencilState.DepthEnable = desc.disableDepthTest ? FALSE : TRUE;
        // Blended surfaces, proxies, the sky and the lighting triangle
        // test against the opaque depth but never write it.
        pso.DepthStencilState.DepthWriteMask =
            (desc.alphaBlend || desc.occlusionProxy || desc.occlusionDebug || desc.background ||
             desc.disableDepthTest)
                ? D3D12_DEPTH_WRITE_MASK_ZERO
                : D3D12_DEPTH_WRITE_MASK_ALL;
        // LESS_EQUAL for proxies/sky: coplanar zero-extent boxes and the
        // far-plane triangle must pass against their own depth.
        pso.DepthStencilState.DepthFunc =
            (desc.occlusionProxy || desc.occlusionDebug || desc.background)
                ? D3D12_COMPARISON_FUNC_LESS_EQUAL
                : D3D12_COMPARISON_FUNC_LESS;
        pso.DSVFormat = toDxgi(desc.depthFormat);
    } else {
        pso.DepthStencilState.DepthEnable = FALSE;
        pso.DSVFormat = DXGI_FORMAT_UNKNOWN;
    }
    pso.DepthStencilState.StencilEnable = FALSE;

    // MRT (colorFormats non-empty) wins over the single colorFormat; no
    // colour format at all = depth-only (shadow passes).
    std::vector<Format> colorFormats;
    if (!desc.colorFormats.empty()) {
        colorFormats = desc.colorFormats;
    } else if (desc.colorFormat != Format::Undefined) {
        colorFormats.push_back(desc.colorFormat);
    }
    if (colorFormats.size() > 8) {
        return Error{"D3D12 backend: at most 8 colour attachments"};
    }
    pso.NumRenderTargets = static_cast<UINT>(colorFormats.size());
    for (std::size_t i = 0; i < colorFormats.size(); ++i) {
        pso.RTVFormats[i] = toDxgi(colorFormats[i]);
    }
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso.SampleDesc.Count = 1;

    auto pipeline = std::unique_ptr<D3D12Pipeline>(new D3D12Pipeline());
    if (HRESULT hr = device.handle()->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&pipeline->pso_));
        FAILED(hr)) {
        return hrError("CreateGraphicsPipelineState", hr);
    }
    pipeline->rootSignature_ = table.rootSignature();
    pipeline->vertexStride_ = desc.vertexStride;
    log::info("Graphics pipeline created (D3D12, {} color attachment(s), format {})",
              colorFormats.size(),
              colorFormats.empty() ? "none" : formatName(colorFormats.front()));
    return std::unique_ptr<Pipeline>(std::move(pipeline));
}

Result<std::unique_ptr<Pipeline>> D3D12Pipeline::createCompute(const Device& deviceBase,
                                                               const ComputePipelineDesc& desc) {
    const D3D12Device& device = dx(deviceBase);
    if (!desc.shader) {
        return Error{"Compute pipeline needs a shader"};
    }
    if (!desc.descriptorTable) {
        return Error{"D3D12 backend: a compute pipeline needs the descriptor table (it owns the "
                     "root signature)"};
    }
    const D3D12DescriptorTable& table = dx(*desc.descriptorTable);

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = table.rootSignature();
    pso.CS = dx(*desc.shader).bytecode();

    auto pipeline = std::unique_ptr<D3D12Pipeline>(new D3D12Pipeline());
    if (HRESULT hr = device.handle()->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline->pso_));
        FAILED(hr)) {
        return hrError("CreateComputePipelineState", hr);
    }
    pipeline->rootSignature_ = table.rootSignature();
    pipeline->compute_ = true;
    log::info("Compute pipeline created (D3D12)");
    return std::unique_ptr<Pipeline>(std::move(pipeline));
}

} // namespace rend::gpu

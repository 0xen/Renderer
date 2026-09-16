// D3D12 backend: the command-list recording surface. Barriers, render
// targets + clears, viewport/scissor, pipeline/table/root-constant binds,
// direct and ExecuteIndirect draws, dispatches, zero fills — with buffer
// resource states tracked per command list (see D3D12Device).
#include "d3d12_types.h"

#include "rend/core/log.h"

#include <algorithm>

namespace rend::gpu {

namespace {

D3D12_RESOURCE_STATES stateOf(ImageLayout layout, bool depth) {
    switch (layout) {
    case ImageLayout::ColorAttachment: return D3D12_RESOURCE_STATE_RENDER_TARGET;
    case ImageLayout::DepthAttachment: return D3D12_RESOURCE_STATE_DEPTH_WRITE;
    case ImageLayout::ShaderReadOnly:
        return depth ? (D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
                     : (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    case ImageLayout::Present: return D3D12_RESOURCE_STATE_PRESENT;
    case ImageLayout::Undefined:
    default: return D3D12_RESOURCE_STATE_COMMON;
    }
}

D3D12_RESOURCE_STATES stateOf(ImageState state, bool depth) {
    switch (state) {
    case ImageState::ColorAttachment: return D3D12_RESOURCE_STATE_RENDER_TARGET;
    case ImageState::DepthAttachment: return D3D12_RESOURCE_STATE_DEPTH_WRITE;
    case ImageState::ShaderRead: return stateOf(ImageLayout::ShaderReadOnly, depth);
    case ImageState::Undefined:
    default: return D3D12_RESOURCE_STATE_COMMON;
    }
}

bool storageAccess(Access access) {
    const auto bits = static_cast<std::uint32_t>(access);
    return (bits & static_cast<std::uint32_t>(Access::ShaderStorageWrite)) != 0 ||
           (bits & static_cast<std::uint32_t>(Access::ShaderStorageRead)) != 0 ||
           (bits & static_cast<std::uint32_t>(Access::AccelerationStructureWrite)) != 0 ||
           (bits & static_cast<std::uint32_t>(Access::TransferWrite)) != 0;
}

void errorOnce(const char* what) {
    static const char* seen[16] = {};
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

constexpr D3D12_RESOURCE_STATES kVertexIndexState =
    D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_INDEX_BUFFER;
constexpr D3D12_RESOURCE_STATES kShaderResourceState =
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

} // namespace

// ------------------------------------------------------------- barriers

void D3D12CommandContext::transition(const Image& imageBase, D3D12_RESOURCE_STATES to) {
    const D3D12Image& image = dx(imageBase);
    const D3D12_RESOURCE_STATES from = image.state();
    if (from == to) {
        return;
    }
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = image.resource();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = from;
    barrier.Transition.StateAfter = to;
    list_->ResourceBarrier(1, &barrier);
    image.setState(to);
}

void D3D12CommandContext::imageBarrier(const Image& image, ImageState /*from*/, ImageState to) {
    // The tracked state replaces `from`; Undefined keeps whatever the
    // image is in (contents are discarded by the following clear).
    if (to == ImageState::Undefined) {
        return;
    }
    transition(image, stateOf(to, isDepthFormat(image.format())));
}

void D3D12CommandContext::memoryBarrier(Stage /*from*/, Stage /*to*/) {
    // Same-queue execution order is implicit; UAV writes need the UAV
    // barrier to become visible to the next pass.
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = nullptr;
    list_->ResourceBarrier(1, &barrier);
}

void D3D12CommandContext::barrier(std::span<const MemoryBarrierDesc> memory,
                                  std::span<const ImageBarrierDesc> images) {
    // Buffer hazards: every storage buffer is a UAV, so one UAV barrier
    // covers shader write -> read; the vertex/index/indirect/copy uses
    // are ordered by the state transitions applied at each command.
    bool uav = false;
    for (const MemoryBarrierDesc& m : memory) {
        if (storageAccess(m.srcAccess) || storageAccess(m.dstAccess)) {
            uav = true;
        }
    }
    if (uav) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = nullptr;
        list_->ResourceBarrier(1, &barrier);
    }
    for (const ImageBarrierDesc& d : images) {
        if (d.image == nullptr || d.newLayout == ImageLayout::Undefined) {
            continue;
        }
        transition(*d.image, stateOf(d.newLayout, isDepthFormat(d.image->format())));
    }
}

void D3D12CommandContext::bufferTransition(const D3D12Buffer& buffer, D3D12_RESOURCE_STATES to,
                                           std::vector<D3D12_RESOURCE_BARRIER>& out) {
    const D3D12_RESOURCE_STATES from = buffer.state();
    if (from == to) {
        return;
    }
    // Explicit even from COMMON: buffers always decay to COMMON between
    // command lists, so the before-state is exact and nothing depends on
    // implicit promotion.
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = buffer.resource();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = from;
    barrier.Transition.StateAfter = to;
    out.push_back(barrier);
    buffer.setState(to);
}

void D3D12CommandContext::applyBufferStates(bool graphics, const D3D12Buffer* indirectArgs,
                                            const D3D12Buffer* indirectCount) {
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    for (const D3D12Buffer* buffer : device_->trackedBuffers()) {
        // Shader access: UAV where any binding writes the buffer, the
        // shader-resource states for the read-only ones (SRV loads).
        D3D12_RESOURCE_STATES want =
            buffer->uavBound() ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : kShaderResourceState;
        if (buffer == indirectArgs || buffer == indirectCount) {
            want = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        } else if (graphics && (buffer == vertexBuffer_ || buffer == indexBuffer_)) {
            want = kVertexIndexState;
        }
        bufferTransition(*buffer, want, barriers);
    }
    if (!barriers.empty()) {
        list_->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
    }
}

void D3D12CommandContext::fillBuffer(const Buffer& bufferBase, std::uint64_t offset,
                                     std::uint64_t size, std::uint32_t value) {
    if (value != 0) {
        errorOnce("fillBuffer supports only zero fills (the engine never fills anything else)");
        return;
    }
    const D3D12Buffer& buffer = dx(bufferBase);
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    bufferTransition(buffer, D3D12_RESOURCE_STATE_COPY_DEST, barriers);
    if (!barriers.empty()) {
        list_->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
    }
    for (std::uint64_t done = 0; done < size;) {
        const std::uint64_t chunk = std::min(size - done, D3D12Device::kZeroBytes);
        list_->CopyBufferRegion(buffer.resource(), offset + done, device_->zeroBuffer(), 0, chunk);
        done += chunk;
    }
}

// ------------------------------------------------------------ rendering

void D3D12CommandContext::beginRendering(const RenderingDesc& desc) {
    std::array<D3D12_CPU_DESCRIPTOR_HANDLE, 8> rtvs{};
    std::uint32_t count = 0;
    for (const ColorTarget& target : desc.colors) {
        if (target.image == nullptr || count == rtvs.size()) {
            continue;
        }
        rtvs[count++] = dx(*target.image).rtv();
    }
    D3D12_CPU_DESCRIPTOR_HANDLE dsv{};
    const bool hasDepth = desc.depth != nullptr && desc.depth->image != nullptr;
    if (hasDepth) {
        dsv = dx(*desc.depth->image).dsv();
    }
    list_->OMSetRenderTargets(count, rtvs.data(), FALSE, hasDepth ? &dsv : nullptr);

    std::uint32_t i = 0;
    for (const ColorTarget& target : desc.colors) {
        if (target.image == nullptr) {
            continue;
        }
        if (target.load == LoadOp::Clear) {
            list_->ClearRenderTargetView(rtvs[i], target.clear.data(), 0, nullptr);
        }
        ++i;
    }
    if (hasDepth && desc.depth->load == LoadOp::Clear) {
        list_->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, desc.depth->clear, 0, 0, nullptr);
    }

    setViewport(0.0f, 0.0f, static_cast<float>(desc.width), static_cast<float>(desc.height));
    setScissor(0, 0, desc.width, desc.height);
}

void D3D12CommandContext::endRendering() {}

void D3D12CommandContext::setViewport(float x, float y, float width, float height) {
    D3D12_VIEWPORT viewport{x, y, width, height, 0.0f, 1.0f};
    list_->RSSetViewports(1, &viewport);
}

void D3D12CommandContext::setScissor(std::int32_t x, std::int32_t y, std::uint32_t width,
                                     std::uint32_t height) {
    D3D12_RECT rect{x, y, static_cast<LONG>(x + static_cast<std::int32_t>(width)),
                    static_cast<LONG>(y + static_cast<std::int32_t>(height))};
    list_->RSSetScissorRects(1, &rect);
}

// ---------------------------------------------------------------- binds

void D3D12CommandContext::bindPipeline(const Pipeline& pipelineBase) {
    const D3D12Pipeline& pipeline = dx(pipelineBase);
    pipeline_ = &pipeline;
    // Root arguments survive pipeline changes as long as the root
    // signature object stays the same (every pipeline built against one
    // table shares it), so it is only re-set when it actually changes.
    if (pipeline.isCompute()) {
        if (computeRootSignature_ != pipeline.rootSignature()) {
            computeRootSignature_ = pipeline.rootSignature();
            list_->SetComputeRootSignature(computeRootSignature_);
        }
    } else {
        if (graphicsRootSignature_ != pipeline.rootSignature()) {
            graphicsRootSignature_ = pipeline.rootSignature();
            list_->SetGraphicsRootSignature(graphicsRootSignature_);
        }
        list_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    }
    list_->SetPipelineState(pipeline.pso());
}

void D3D12CommandContext::bindDescriptorTable(const Pipeline& pipeline,
                                              const DescriptorTable& tableBase) {
    const D3D12DescriptorTable& table = dx(tableBase);
    if (table_ != &table) {
        ID3D12DescriptorHeap* heaps[2] = {table.resourceHeap(), table.samplerHeap()};
        list_->SetDescriptorHeaps(2, heaps);
        table_ = &table;
    }
    if (pipeline.isCompute()) {
        list_->SetComputeRootDescriptorTable(2, table.resourceTableStart());
        list_->SetComputeRootDescriptorTable(3, table.samplerTableStart());
    } else {
        list_->SetGraphicsRootDescriptorTable(2, table.resourceTableStart());
        list_->SetGraphicsRootDescriptorTable(3, table.samplerTableStart());
    }
}

void D3D12CommandContext::pushConstants(const Pipeline& pipeline, const void* data,
                                        std::uint32_t bytes) {
    const UINT dwords = std::min<UINT>(bytes / 4, D3D12DescriptorTable::kPushDwords);
    if (pipeline.isCompute()) {
        list_->SetComputeRoot32BitConstants(0, dwords, data, 0);
    } else {
        list_->SetGraphicsRoot32BitConstants(0, dwords, data, 0);
    }
}

void D3D12CommandContext::bindVertexBuffer(const Buffer& bufferBase, std::uint64_t offset) {
    const D3D12Buffer& buffer = dx(bufferBase);
    vertexBuffer_ = &buffer;
    const std::uint32_t stride = pipeline_ ? pipeline_->vertexStride() : 0;
    if (stride == 0) {
        errorOnce("bindVertexBuffer before a pipeline with vertex input was bound");
    }
    D3D12_VERTEX_BUFFER_VIEW view{};
    view.BufferLocation = buffer.gpuAddress() + offset;
    view.SizeInBytes = static_cast<UINT>(buffer.size() - offset);
    view.StrideInBytes = stride;
    list_->IASetVertexBuffers(0, 1, &view);
}

void D3D12CommandContext::bindIndexBuffer(const Buffer& bufferBase, std::uint64_t offset) {
    const D3D12Buffer& buffer = dx(bufferBase);
    indexBuffer_ = &buffer;
    D3D12_INDEX_BUFFER_VIEW view{};
    view.BufferLocation = buffer.gpuAddress() + offset;
    view.SizeInBytes = static_cast<UINT>(buffer.size() - offset);
    view.Format = DXGI_FORMAT_R32_UINT;
    list_->IASetIndexBuffer(&view);
}

void D3D12CommandContext::setBaseInstance(std::uint32_t firstInstance) {
    list_->SetGraphicsRoot32BitConstant(1, firstInstance, 0);
}

// ---------------------------------------------------------------- draws

void D3D12CommandContext::draw(std::uint32_t vertexCount, std::uint32_t instanceCount,
                               std::uint32_t firstVertex, std::uint32_t firstInstance) {
    applyBufferStates(true, nullptr, nullptr);
    setBaseInstance(firstInstance);
    list_->DrawInstanced(vertexCount, instanceCount, firstVertex, firstInstance);
}

void D3D12CommandContext::drawIndexed(std::uint32_t indexCount, std::uint32_t instanceCount,
                                      std::uint32_t firstIndex, std::int32_t vertexOffset,
                                      std::uint32_t firstInstance) {
    applyBufferStates(true, nullptr, nullptr);
    setBaseInstance(firstInstance);
    list_->DrawIndexedInstanced(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

void D3D12CommandContext::drawIndirect(const Buffer& bufferBase, std::uint64_t offset,
                                       std::uint32_t drawCount, std::uint32_t stride) {
    const D3D12Buffer& buffer = dx(bufferBase);
    auto signature = device_->drawSignature(stride == 0 ? 16 : stride);
    if (!signature) {
        errorOnce("drawIndirect: no command signature");
        return;
    }
    applyBufferStates(true, &buffer, nullptr);
    setBaseInstance(0);
    list_->ExecuteIndirect(signature.value(), drawCount, buffer.resource(), offset, nullptr, 0);
}

void D3D12CommandContext::drawIndexedIndirect(const Buffer& bufferBase, std::uint64_t offset,
                                              std::uint32_t drawCount, std::uint32_t stride) {
    if (table_ == nullptr) {
        errorOnce("drawIndexedIndirect without a bound descriptor table");
        return;
    }
    if (stride != 0 && stride != D3D12DescriptorTable::kIndirectStride) {
        errorOnce("drawIndexedIndirect: records must be DrawIndexedIndirect (24 bytes)");
        return;
    }
    const D3D12Buffer& buffer = dx(bufferBase);
    applyBufferStates(true, &buffer, nullptr);
    list_->ExecuteIndirect(table_->drawIndexedSignature(), drawCount, buffer.resource(), offset,
                           nullptr, 0);
}

void D3D12CommandContext::drawIndexedIndirectCount(const Buffer& bufferBase, std::uint64_t offset,
                                                   const Buffer& countBase,
                                                   std::uint64_t countOffset,
                                                   std::uint32_t maxDrawCount,
                                                   std::uint32_t stride) {
    if (table_ == nullptr) {
        errorOnce("drawIndexedIndirectCount without a bound descriptor table");
        return;
    }
    if (stride != 0 && stride != D3D12DescriptorTable::kIndirectStride) {
        errorOnce("drawIndexedIndirectCount: records must be DrawIndexedIndirect (24 bytes)");
        return;
    }
    const D3D12Buffer& buffer = dx(bufferBase);
    const D3D12Buffer& count = dx(countBase);
    applyBufferStates(true, &buffer, &count);
    list_->ExecuteIndirect(table_->drawIndexedSignature(), maxDrawCount, buffer.resource(), offset,
                           count.resource(), countOffset);
}

void D3D12CommandContext::dispatch(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    applyBufferStates(false, nullptr, nullptr);
    list_->Dispatch(x, y, z);
}

} // namespace rend::gpu

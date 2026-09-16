// D3D12 backend: the command-list recording surface. Skeleton scope —
// barriers, render targets + clears, viewport/scissor. Draw/bind/dispatch
// verbs are logged once and skipped until the scene path (slice 3).
#include "d3d12_types.h"

#include "rend/core/log.h"

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

void unsupported(const char* verb) {
    static const char* names[16] = {};
    for (int i = 0; i < 16; ++i) {
        if (names[i] == verb) {
            return;
        }
        if (names[i] == nullptr) {
            names[i] = verb;
            log::warn("D3D12 backend: CommandContext::{} not implemented yet (skipped)", verb);
            return;
        }
    }
}

} // namespace

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

void D3D12CommandContext::fillBuffer(const Buffer& /*buffer*/, std::uint64_t /*offset*/,
                                     std::uint64_t /*size*/, std::uint32_t /*value*/) {
    unsupported("fillBuffer");
}

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

void D3D12CommandContext::bindPipeline(const Pipeline& /*pipeline*/) { unsupported("bindPipeline"); }

void D3D12CommandContext::bindDescriptorTable(const Pipeline& /*pipeline*/,
                                              const DescriptorTable& /*table*/) {
    unsupported("bindDescriptorTable");
}

void D3D12CommandContext::pushConstants(const Pipeline& /*pipeline*/, const void* /*data*/,
                                        std::uint32_t /*bytes*/) {
    unsupported("pushConstants");
}

void D3D12CommandContext::bindVertexBuffer(const Buffer& /*buffer*/, std::uint64_t /*offset*/) {
    unsupported("bindVertexBuffer");
}

void D3D12CommandContext::bindIndexBuffer(const Buffer& /*buffer*/, std::uint64_t /*offset*/) {
    unsupported("bindIndexBuffer");
}

void D3D12CommandContext::draw(std::uint32_t /*vertexCount*/, std::uint32_t /*instanceCount*/,
                               std::uint32_t /*firstVertex*/, std::uint32_t /*firstInstance*/) {
    unsupported("draw");
}

void D3D12CommandContext::drawIndexed(std::uint32_t /*indexCount*/, std::uint32_t /*instanceCount*/,
                                      std::uint32_t /*firstIndex*/, std::int32_t /*vertexOffset*/,
                                      std::uint32_t /*firstInstance*/) {
    unsupported("drawIndexed");
}

void D3D12CommandContext::drawIndirect(const Buffer& /*buffer*/, std::uint64_t /*offset*/,
                                       std::uint32_t /*drawCount*/, std::uint32_t /*stride*/) {
    unsupported("drawIndirect");
}

void D3D12CommandContext::drawIndexedIndirect(const Buffer& /*buffer*/, std::uint64_t /*offset*/,
                                              std::uint32_t /*drawCount*/,
                                              std::uint32_t /*stride*/) {
    unsupported("drawIndexedIndirect");
}

void D3D12CommandContext::drawIndexedIndirectCount(const Buffer& /*buffer*/,
                                                   std::uint64_t /*offset*/,
                                                   const Buffer& /*count*/,
                                                   std::uint64_t /*countOffset*/,
                                                   std::uint32_t /*maxDrawCount*/,
                                                   std::uint32_t /*stride*/) {
    unsupported("drawIndexedIndirectCount");
}

void D3D12CommandContext::dispatch(std::uint32_t /*x*/, std::uint32_t /*y*/, std::uint32_t /*z*/) {
    unsupported("dispatch");
}

} // namespace rend::gpu

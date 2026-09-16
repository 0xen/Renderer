// D3D12 backend: load-time uploads. TransferContext batches buffer copies
// on the copy queue; TextureUploader builds mipmapped textures on the
// direct queue (mips generated on the CPU — D3D12 has no blit).
#include "d3d12_types.h"

#include "rend/core/log.h"
#include "rend/core/profile.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>

namespace rend::gpu {

namespace {

constexpr std::uint64_t kInitialStagingBytes = 4ull * 1024 * 1024;
constexpr DWORD kFenceTimeoutMs = 30000;

Result<void> waitFence(ID3D12Fence* fence, std::uint64_t value, HANDLE event, const char* what) {
    if (fence->GetCompletedValue() >= value) {
        return {};
    }
    if (HRESULT hr = fence->SetEventOnCompletion(value, event); FAILED(hr)) {
        return hrError("SetEventOnCompletion", hr);
    }
    if (WaitForSingleObject(event, kFenceTimeoutMs) != WAIT_OBJECT_0) {
        return Error{std::format("{} fence never signalled; the GPU appears stalled", what)};
    }
    return {};
}

Result<void> createSubmitObjects(const D3D12Device& device, D3D12_COMMAND_LIST_TYPE type,
                                 ComPtr<ID3D12CommandAllocator>& allocator,
                                 ComPtr<ID3D12GraphicsCommandList>& list, ComPtr<ID3D12Fence>& fence,
                                 HANDLE& event) {
    if (HRESULT hr = device.handle()->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator));
        FAILED(hr)) {
        return hrError("CreateCommandAllocator", hr);
    }
    if (HRESULT hr = device.handle()->CreateCommandList(0, type, allocator.Get(), nullptr,
                                                        IID_PPV_ARGS(&list));
        FAILED(hr)) {
        return hrError("CreateCommandList", hr);
    }
    list->Close(); // created open; every use resets it
    if (HRESULT hr = device.handle()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        FAILED(hr)) {
        return hrError("CreateFence", hr);
    }
    event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) {
        return Error{"CreateEvent failed"};
    }
    return {};
}

// sRGB <-> linear for the CPU mip chain (the Vulkan blit filters sRGB
// content in linear space).
float srgbToLinear(std::uint8_t v) {
    const float c = static_cast<float>(v) / 255.0f;
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

std::uint8_t linearToSrgb(float c) {
    c = std::clamp(c, 0.0f, 1.0f);
    const float s = c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
    return static_cast<std::uint8_t>(std::lround(s * 255.0f));
}

// One mip level down: 2x2 box filter (clamped at odd edges), RGB in
// linear space for sRGB content, alpha linear either way.
void downsample(const std::uint8_t* src, std::uint32_t srcW, std::uint32_t srcH, std::uint8_t* dst,
                std::uint32_t dstW, std::uint32_t dstH, bool srgb, const float* toLinear) {
    for (std::uint32_t y = 0; y < dstH; ++y) {
        const std::uint32_t y0 = std::min(2 * y, srcH - 1);
        const std::uint32_t y1 = std::min(2 * y + 1, srcH - 1);
        for (std::uint32_t x = 0; x < dstW; ++x) {
            const std::uint32_t x0 = std::min(2 * x, srcW - 1);
            const std::uint32_t x1 = std::min(2 * x + 1, srcW - 1);
            const std::uint8_t* p[4] = {src + 4 * (y0 * srcW + x0), src + 4 * (y0 * srcW + x1),
                                        src + 4 * (y1 * srcW + x0), src + 4 * (y1 * srcW + x1)};
            std::uint8_t* out = dst + 4 * (y * dstW + x);
            for (int c = 0; c < 3; ++c) {
                if (srgb) {
                    const float sum = toLinear[p[0][c]] + toLinear[p[1][c]] + toLinear[p[2][c]] +
                                      toLinear[p[3][c]];
                    out[c] = linearToSrgb(sum * 0.25f);
                } else {
                    out[c] = static_cast<std::uint8_t>((p[0][c] + p[1][c] + p[2][c] + p[3][c] + 2) / 4);
                }
            }
            out[3] = static_cast<std::uint8_t>((p[0][3] + p[1][3] + p[2][3] + p[3][3] + 2) / 4);
        }
    }
}

} // namespace

// --------------------------------------------------------- TransferContext

Result<std::unique_ptr<TransferContext>> D3D12TransferContext::create(const Device& deviceBase) {
    const D3D12Device& device = dx(deviceBase);
    auto context = std::unique_ptr<D3D12TransferContext>(new D3D12TransferContext());
    context->device_ = &device;
    if (auto r = createSubmitObjects(device, D3D12_COMMAND_LIST_TYPE_COPY, context->allocator_,
                                     context->list_, context->fence_, context->fenceEvent_);
        !r) {
        return r.error();
    }
    return std::unique_ptr<TransferContext>(std::move(context));
}

Result<void> D3D12TransferContext::ensureStagingCapacity(std::uint64_t required) {
    if (staging_ && staging_->size() >= required) {
        return {};
    }
    std::uint64_t capacity = staging_ ? staging_->size() : kInitialStagingBytes;
    while (capacity < required) {
        capacity *= 2;
    }
    // No usage bits: a plain upload-heap buffer (always a copy source).
    auto grown = D3D12Buffer::create(*device_, {.size = capacity,
                                                .usage = 0,
                                                .location = MemoryLocation::HostVisible});
    if (!grown) {
        return grown.error();
    }
    if (staging_ && stagingUsed_ > 0) {
        std::memcpy(grown.value()->mapped(), staging_->mapped(), stagingUsed_);
    }
    staging_ = std::move(grown).value();
    return {};
}

Result<void> D3D12TransferContext::stage(const Buffer& dst, std::uint64_t dstOffset,
                                         const void* data, std::uint64_t size) {
    if (size == 0) {
        return {};
    }
    if (dstOffset + size > dst.size()) {
        return Error{std::format("Staged copy overruns destination ({} + {} > {})", dstOffset,
                                 size, dst.size())};
    }
    if (auto r = ensureStagingCapacity(stagingUsed_ + size); !r) {
        return r;
    }
    std::memcpy(static_cast<std::byte*>(staging_->mapped()) + stagingUsed_, data, size);
    pending_.push_back({.dst = &dst, .srcOffset = stagingUsed_, .dstOffset = dstOffset, .size = size});
    stagingUsed_ += size;
    return {};
}

Result<void> D3D12TransferContext::flush() {
    if (pending_.empty()) {
        return {};
    }
    if (HRESULT hr = allocator_->Reset(); FAILED(hr)) {
        return hrError("CommandAllocator::Reset (transfer)", hr);
    }
    if (HRESULT hr = list_->Reset(allocator_.Get(), nullptr); FAILED(hr)) {
        return hrError("CommandList::Reset (transfer)", hr);
    }
    // Destinations sit in COMMON between command lists (buffers decay),
    // which is what the copy queue requires.
    for (const PendingCopy& copy : pending_) {
        list_->CopyBufferRegion(dx(*copy.dst).resource(), copy.dstOffset, dx(*staging_).resource(),
                                copy.srcOffset, copy.size);
    }
    if (HRESULT hr = list_->Close(); FAILED(hr)) {
        return hrError("CommandList::Close (transfer)", hr);
    }
    ID3D12CommandList* lists[] = {list_.Get()};
    device_->copyQueue()->ExecuteCommandLists(1, lists);
    const std::uint64_t value = ++fenceValue_;
    device_->copyQueue()->Signal(fence_.Get(), value);
    if (auto r = waitFence(fence_.Get(), value, fenceEvent_, "Transfer"); !r) {
        return r;
    }
    log::trace("Transfer flushed: {} copies, {} bytes", pending_.size(), stagingUsed_);
    pending_.clear();
    stagingUsed_ = 0;
    return {};
}

// --------------------------------------------------------- TextureUploader

Result<std::unique_ptr<TextureUploader>> D3D12TextureUploader::create(const Device& deviceBase) {
    const D3D12Device& device = dx(deviceBase);
    auto uploader = std::unique_ptr<D3D12TextureUploader>(new D3D12TextureUploader());
    uploader->device_ = &device;
    if (auto r = createSubmitObjects(device, D3D12_COMMAND_LIST_TYPE_DIRECT, uploader->allocator_,
                                     uploader->list_, uploader->fence_, uploader->fenceEvent_);
        !r) {
        return r.error();
    }
    return std::unique_ptr<TextureUploader>(std::move(uploader));
}

D3D12TextureUploader::~D3D12TextureUploader() {
    if (staging_ && stagingMapped_) {
        staging_->Unmap(0, nullptr);
    }
    if (fenceEvent_) {
        CloseHandle(fenceEvent_);
    }
}

Result<void> D3D12TextureUploader::ensureStagingCapacity(std::uint64_t required) {
    if (staging_ && stagingBytes_ >= required) {
        return {};
    }
    if (staging_ && stagingMapped_) {
        staging_->Unmap(0, nullptr);
        stagingMapped_ = nullptr;
    }
    staging_.Reset();
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = std::bit_ceil(required);
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (HRESULT hr = device_->handle()->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&staging_));
        FAILED(hr)) {
        return hrError("CreateCommittedResource (texture staging)", hr);
    }
    const D3D12_RANGE noRead{0, 0};
    if (HRESULT hr = staging_->Map(0, &noRead, &stagingMapped_); FAILED(hr)) {
        return hrError("Map (texture staging)", hr);
    }
    stagingBytes_ = desc.Width;
    return {};
}

Result<std::unique_ptr<Image>> D3D12TextureUploader::uploadMips(const ImageDesc& desc,
                                                                std::span<const MipSource> mips) {
    auto imageResult = Image::create(*device_, desc);
    if (!imageResult) {
        return imageResult.error();
    }
    auto image = std::move(imageResult).value();
    const D3D12Image& target = dx(*image);
    const UINT mipCount = static_cast<UINT>(mips.size());

    const D3D12_RESOURCE_DESC resourceDesc = target.resource()->GetDesc();
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(mipCount);
    std::vector<UINT> rows(mipCount);
    std::vector<UINT64> rowBytes(mipCount);
    UINT64 total = 0;
    device_->handle()->GetCopyableFootprints(&resourceDesc, 0, mipCount, 0, layouts.data(),
                                             rows.data(), rowBytes.data(), &total);
    if (auto r = ensureStagingCapacity(total); !r) {
        return r.error();
    }
    auto* staging = static_cast<std::byte*>(stagingMapped_);
    for (UINT m = 0; m < mipCount; ++m) {
        const MipSource& src = mips[m];
        const std::uint64_t copyBytes = std::min<std::uint64_t>(src.rowBytes, rowBytes[m]);
        const UINT rowCount = std::min<UINT>(src.rows, rows[m]);
        for (UINT row = 0; row < rowCount; ++row) {
            std::memcpy(staging + layouts[m].Offset + std::uint64_t{row} * layouts[m].Footprint.RowPitch,
                        src.pixels + std::uint64_t{row} * src.rowBytes, copyBytes);
        }
    }

    if (HRESULT hr = allocator_->Reset(); FAILED(hr)) {
        return hrError("CommandAllocator::Reset (texture)", hr);
    }
    if (HRESULT hr = list_->Reset(allocator_.Get(), nullptr); FAILED(hr)) {
        return hrError("CommandList::Reset (texture)", hr);
    }
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = target.resource();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = target.state();
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    list_->ResourceBarrier(1, &barrier);
    for (UINT m = 0; m < mipCount; ++m) {
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = target.resource();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = m;
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = staging_.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = layouts[m];
        list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    const D3D12_RESOURCE_STATES sampled = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                                          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = sampled;
    list_->ResourceBarrier(1, &barrier);
    target.setState(sampled);
    if (HRESULT hr = list_->Close(); FAILED(hr)) {
        return hrError("CommandList::Close (texture)", hr);
    }
    ID3D12CommandList* lists[] = {list_.Get()};
    device_->graphicsQueue()->ExecuteCommandLists(1, lists);
    const std::uint64_t value = ++fenceValue_;
    device_->graphicsQueue()->Signal(fence_.Get(), value);
    if (auto r = waitFence(fence_.Get(), value, fenceEvent_, "Texture upload"); !r) {
        return r.error();
    }
    return image;
}

Result<std::unique_ptr<Image>> D3D12TextureUploader::upload(std::uint32_t width,
                                                            std::uint32_t height,
                                                            const void* rgba8, bool srgb) {
    REND_PROFILE_ZONE("TextureUpload");
    if (width == 0 || height == 0 || !rgba8) {
        return Error{"Texture upload needs non-empty pixels"};
    }
    const std::uint32_t mipLevels = std::bit_width(std::max(width, height));

    static float toLinear[256];
    static bool tableReady = false;
    if (!tableReady) {
        for (int i = 0; i < 256; ++i) {
            toLinear[i] = srgbToLinear(static_cast<std::uint8_t>(i));
        }
        tableReady = true;
    }

    // Level 0 is the caller's buffer; every further level is filtered
    // from the previous one.
    std::vector<std::vector<std::uint8_t>> levels(mipLevels);
    std::vector<MipSource> sources(mipLevels);
    sources[0] = {static_cast<const std::byte*>(rgba8), width * 4, height};
    std::uint32_t w = width;
    std::uint32_t h = height;
    const std::uint8_t* previous = static_cast<const std::uint8_t*>(rgba8);
    for (std::uint32_t m = 1; m < mipLevels; ++m) {
        const std::uint32_t nw = std::max(w / 2, 1u);
        const std::uint32_t nh = std::max(h / 2, 1u);
        levels[m].resize(std::size_t{nw} * nh * 4);
        downsample(previous, w, h, levels[m].data(), nw, nh, srgb, toLinear);
        sources[m] = {reinterpret_cast<const std::byte*>(levels[m].data()), nw * 4, nh};
        previous = levels[m].data();
        w = nw;
        h = nh;
    }

    return uploadMips({.width = width,
                       .height = height,
                       .format = srgb ? kFormatR8G8B8A8Srgb : kFormatR8G8B8A8Unorm,
                       .usage = kImageUsageSampled | kImageUsageTransferDst,
                       .mipLevels = mipLevels},
                      sources);
}

Result<std::unique_ptr<Image>> D3D12TextureUploader::uploadCompressed(Format format,
                                                                      const CompressedMip* mips,
                                                                      std::uint32_t mipCount,
                                                                      const void* bytes,
                                                                      std::uint64_t byteSize) {
    REND_PROFILE_ZONE("TextureUpload");
    if (mipCount == 0 || !bytes || byteSize == 0) {
        return Error{"Compressed upload needs non-empty mips"};
    }
    // BC7: 4x4 blocks of 16 bytes, rows are block rows.
    std::vector<MipSource> sources(mipCount);
    for (std::uint32_t m = 0; m < mipCount; ++m) {
        if (mips[m].byteOffset + mips[m].byteLength > byteSize) {
            return Error{std::format("Compressed mip {} overruns the payload", m)};
        }
        const std::uint32_t blocksWide = std::max(1u, (mips[m].width + 3) / 4);
        const std::uint32_t blocksHigh = std::max(1u, (mips[m].height + 3) / 4);
        sources[m] = {static_cast<const std::byte*>(bytes) + mips[m].byteOffset, blocksWide * 16,
                      blocksHigh};
    }
    return uploadMips({.width = mips[0].width,
                       .height = mips[0].height,
                       .format = format,
                       .usage = kImageUsageSampled | kImageUsageTransferDst,
                       .mipLevels = mipCount},
                      sources);
}

} // namespace rend::gpu

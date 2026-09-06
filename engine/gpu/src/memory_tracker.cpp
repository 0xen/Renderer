#include "rend/gpu/memory_tracker.h"

#include <atomic>

namespace rend::gpu {

namespace {

std::atomic<std::uint64_t> gBytes[MemoryTracker::kKindCount];
std::atomic<std::uint32_t> gCounts[MemoryTracker::kKindCount];

} // namespace

void MemoryTracker::onAlloc(Kind kind, std::uint64_t bytes) {
    const auto i = static_cast<std::uint32_t>(kind);
    gBytes[i].fetch_add(bytes, std::memory_order_relaxed);
    gCounts[i].fetch_add(1, std::memory_order_relaxed);
}

void MemoryTracker::onFree(Kind kind, std::uint64_t bytes) {
    const auto i = static_cast<std::uint32_t>(kind);
    gBytes[i].fetch_sub(bytes, std::memory_order_relaxed);
    gCounts[i].fetch_sub(1, std::memory_order_relaxed);
}

MemoryTracker::Snapshot MemoryTracker::snapshot() {
    Snapshot out;
    for (std::uint32_t i = 0; i < kKindCount; ++i) {
        out.bytes[i] = gBytes[i].load(std::memory_order_relaxed);
        out.counts[i] = gCounts[i].load(std::memory_order_relaxed);
    }
    return out;
}

} // namespace rend::gpu

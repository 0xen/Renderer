#pragma once

#include <cstdint>

namespace rend::gpu {

// Process-wide ledger of GPU device memory. Every vkAllocateMemory in the
// gpu layer reports here on construction and destruction — Buffer and
// Image are the only two allocation sites (pools, staging, acceleration
// structures and per-slot tables all compose out of Buffers), so hooking
// those two covers every byte. Counters live in the engine DLL (methods
// are exported, deliberately not inline): the viewer reads one shared
// truth through snapshot() for its debug panel, the way the FPS graph
// reads frame timing.
class MemoryTracker {
public:
    enum class Kind : std::uint32_t {
        DeviceBuffer = 0, // VRAM buffers (geometry pool, AS storage, ...)
        HostBuffer = 1,   // host-visible mapped buffers (staging, tables)
        Image = 2,        // textures, attachments, shadow maps, probes
    };
    static constexpr std::uint32_t kKindCount = 3;

    struct Snapshot {
        std::uint64_t bytes[kKindCount] = {};
        std::uint32_t counts[kKindCount] = {};
        std::uint64_t totalBytes() const {
            std::uint64_t sum = 0;
            for (std::uint64_t b : bytes) {
                sum += b;
            }
            return sum;
        }
        std::uint32_t totalCount() const {
            std::uint32_t sum = 0;
            for (std::uint32_t c : counts) {
                sum += c;
            }
            return sum;
        }
    };

    // Thread-safe (atomics): buffers are created from the main thread and
    // destroyed via deferred reclaim, images from load paths — no ordering
    // requirements beyond per-resource pairing.
    static void onAlloc(Kind kind, std::uint64_t bytes);
    static void onFree(Kind kind, std::uint64_t bytes);
    static Snapshot snapshot();
};

} // namespace rend::gpu

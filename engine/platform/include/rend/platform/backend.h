#pragma once

#include "rend/core/result.h"
#include "rend/platform/events.h"
#include "rend/platform/presentation_target.h"
#include "rend/platform/types.h"

#include <memory>
#include <vector>

namespace rend::platform {

enum class BackendKind {
    SDL3,
    // Win32, — future second backend proving the abstraction
};

// One implementation per windowing system. The interface never leaks
// backend types (no SDL/Win32 headers cross this boundary).
class IPlatformBackend {
public:
    virtual ~IPlatformBackend() = default;

    virtual Result<void> initialize() = 0;
    virtual void shutdown() = 0;

    virtual Result<std::unique_ptr<PresentationTarget>> createTarget(const TargetDesc& desc) = 0;

    // Drains pending OS events, translated to portable Event values.
    virtual std::vector<Event> pumpEvents() = 0;

    // --- Vulkan seam (the only place platform touches Vulkan; implemented
    // next milestone): instance extensions needed to present to this
    // backend's targets, and surface creation. Kept out of the interface
    // until the gpu layer exists, so nothing here depends on Vulkan headers.
};

Result<std::unique_ptr<IPlatformBackend>> createBackend(BackendKind kind);

} // namespace rend::platform

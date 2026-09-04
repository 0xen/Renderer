#pragma once

#include "rend/platform/types.h"

namespace rend::platform {

// A surface that can be presented to: a window today; conceptually also a
// raw display on platforms that support it. Backends own the concrete type.
class PresentationTarget {
public:
    virtual ~PresentationTarget() = default;

    PresentationTarget(const PresentationTarget&) = delete;
    PresentationTarget& operator=(const PresentationTarget&) = delete;

    virtual Extent2D sizeInPixels() const = 0;
    virtual WindowStyle style() const = 0;

protected:
    PresentationTarget() = default;
};

} // namespace rend::platform

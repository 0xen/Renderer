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

    // Resizes the window. An app that has to grow a window must come
    // through here rather than resizing the OS window itself: the backend
    // holds the size the windowing system answers size queries with, and
    // on Windows a borderless, non-resizable window's client rect is
    // pinned to it (SDL answers WM_NCCALCSIZE with the size it knows). A
    // raw SetWindowPos then grows the window rect while the client rect —
    // which is what DWM composites — stays at the size the backend still
    // believes, so the window can shrink but never grow past its creation
    // size. Position is the app's own business; only the size is here.
    virtual void setSize(Extent2D size) = 0;

protected:
    PresentationTarget() = default;
};

} // namespace rend::platform

#pragma once

#include "rend/platform/types.h"

namespace rend::platform {

// Small portable key set; grows as milestones need it.
enum class Key {
    Unknown,
    Escape,
    Space,
    Enter,
    F11,
};

struct Event {
    enum class Type {
        CloseRequested,
        Resized,
        KeyDown,
        KeyUp,
    };

    Type type;
    Extent2D size{}; // valid for Resized
    Key key = Key::Unknown; // valid for KeyDown/KeyUp
};

} // namespace rend::platform

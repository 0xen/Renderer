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
    W,
    A,
    S,
    D,
    Q,
    E,
    G,
    // Number row 1-9: the viewer binds them to animation-clip selection.
    Num1,
    Num2,
    Num3,
    Num4,
    Num5,
    Num6,
    Num7,
    Num8,
    Num9,
    LeftShift,
    LeftCtrl,
    Up,
    Down,
    Left,
    Right,
    Count, // one past the last key: sizes held-key tables
};

enum class MouseButton {
    Unknown,
    Left,
    Right,
    Middle,
};

struct Event {
    enum class Type {
        CloseRequested,
        Resized,
        KeyDown,
        KeyUp,
        MouseMoved,
        MouseButtonDown,
        MouseButtonUp,
        MouseWheel,
    };

    Type type;
    Extent2D size{};        // valid for Resized
    Key key = Key::Unknown; // valid for KeyDown/KeyUp
    MouseButton button = MouseButton::Unknown; // valid for MouseButtonDown/Up
    float mouseX = 0.0f;      // window coords, valid for MouseMoved/Button*
    float mouseY = 0.0f;
    float mouseDeltaX = 0.0f; // relative motion, valid for MouseMoved
    float mouseDeltaY = 0.0f;
    float wheelDelta = 0.0f;  // valid for MouseWheel (positive = away)
};

} // namespace rend::platform

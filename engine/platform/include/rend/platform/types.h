#pragma once

#include <cstdint>
#include <string>

namespace rend::platform {

struct Extent2D {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

// How a presentation target appears on screen. Borderless styles are
// first-class: launcher-style UIs draw every pixel of their own chrome.
enum class WindowStyle {
    Decorated,             // regular OS window with title bar and border
    Borderless,            // undecorated, opaque
    BorderlessTransparent, // undecorated with per-pixel alpha (driver-dependent for Vulkan)
    FullscreenBorderless,  // covers the monitor at desktop resolution
    FullscreenExclusive,   // owns the display mode (VK_EXT_full_screen_exclusive later)
};

struct TargetDesc {
    WindowStyle style = WindowStyle::Decorated;
    Extent2D size{1280, 720};
    std::string title = "Renderer";
    int monitorIndex = 0; // which monitor to center on / fill
};

} // namespace rend::platform

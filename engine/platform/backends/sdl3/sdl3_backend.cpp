#include "rend/core/log.h"
#include "rend/platform/backend.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace rend::platform {

namespace {

Key translateKey(SDL_Keycode code) {
    switch (code) {
    case SDLK_ESCAPE: return Key::Escape;
    case SDLK_SPACE: return Key::Space;
    case SDLK_RETURN: return Key::Enter;
    case SDLK_F11: return Key::F11;
    case SDLK_W: return Key::W;
    case SDLK_A: return Key::A;
    case SDLK_S: return Key::S;
    case SDLK_D: return Key::D;
    case SDLK_Q: return Key::Q;
    case SDLK_E: return Key::E;
    case SDLK_G: return Key::G;
    case SDLK_LSHIFT: return Key::LeftShift;
    case SDLK_LCTRL: return Key::LeftCtrl;
    case SDLK_UP: return Key::Up;
    case SDLK_DOWN: return Key::Down;
    case SDLK_LEFT: return Key::Left;
    case SDLK_RIGHT: return Key::Right;
    default: return Key::Unknown;
    }
}

MouseButton translateButton(Uint8 button) {
    switch (button) {
    case SDL_BUTTON_LEFT: return MouseButton::Left;
    case SDL_BUTTON_RIGHT: return MouseButton::Right;
    case SDL_BUTTON_MIDDLE: return MouseButton::Middle;
    default: return MouseButton::Unknown;
    }
}

class Sdl3Target final : public PresentationTarget {
public:
    Sdl3Target(SDL_Window* window, WindowStyle style) : window_(window), style_(style) {}

    ~Sdl3Target() override {
        if (window_) {
            SDL_DestroyWindow(window_);
        }
    }

    Extent2D sizeInPixels() const override {
        int w = 0, h = 0;
        SDL_GetWindowSizeInPixels(window_, &w, &h);
        return {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h)};
    }

    WindowStyle style() const override { return style_; }

    SDL_Window* handle() const { return window_; }

private:
    SDL_Window* window_ = nullptr;
    WindowStyle style_;
};

class Sdl3Backend final : public IPlatformBackend {
public:
    ~Sdl3Backend() override { shutdown(); }

    Result<void> initialize() override {
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            return Error{std::string("SDL_Init failed: ") + SDL_GetError()};
        }
        initialized_ = true;
        log::info("SDL3 backend initialized (SDL {}.{}.{})", SDL_MAJOR_VERSION, SDL_MINOR_VERSION,
                  SDL_MICRO_VERSION);
        return {};
    }

    void shutdown() override {
        if (initialized_) {
            SDL_Quit();
            initialized_ = false;
            log::info("SDL3 backend shut down");
        }
    }

    Result<std::unique_ptr<PresentationTarget>> createTarget(const TargetDesc& desc) override {
        SDL_WindowFlags flags = SDL_WINDOW_VULKAN;
        switch (desc.style) {
        case WindowStyle::Decorated:
            flags |= SDL_WINDOW_RESIZABLE;
            break;
        case WindowStyle::Borderless:
            flags |= SDL_WINDOW_BORDERLESS;
            break;
        case WindowStyle::BorderlessTransparent:
            flags |= SDL_WINDOW_BORDERLESS | SDL_WINDOW_TRANSPARENT;
            break;
        case WindowStyle::FullscreenBorderless:
            flags |= SDL_WINDOW_FULLSCREEN | SDL_WINDOW_BORDERLESS;
            break;
        case WindowStyle::FullscreenExclusive:
            return Error{"FullscreenExclusive is not implemented yet (planned milestone)"};
        }

        SDL_Window* window = SDL_CreateWindow(desc.title.c_str(), static_cast<int>(desc.size.width),
                                              static_cast<int>(desc.size.height), flags);
        if (!window) {
            return Error{std::string("SDL_CreateWindow failed: ") + SDL_GetError()};
        }

#ifdef _WIN32
        // Transparency spike: SDL's SDL_WINDOW_TRANSPARENT only enables DWM
        // blur-behind, which is not enough on every driver (AMD/NVIDIA
        // present through a DXGI flip swapchain that ignores alpha). The
        // documented workaround (SDL issue #15751) is to add layered-window
        // ex-styles. REND_TRANSPARENT_EXSTYLE selects the experiment:
        // none | layered | layered+noredir | noredir | colorkey.
        if (desc.style == WindowStyle::BorderlessTransparent) {
            HWND hwnd = static_cast<HWND>(SDL_GetPointerProperty(
                SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
            const char* mode = SDL_getenv("REND_TRANSPARENT_EXSTYLE");
            const std::string exstyle = mode ? mode : "layered";
            if (hwnd && exstyle != "none") {
                LONG_PTR ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
                if (exstyle.find("layered") != std::string::npos || exstyle == "colorkey") {
                    ex |= WS_EX_LAYERED;
                }
                if (exstyle.find("noredir") != std::string::npos) {
                    ex |= WS_EX_NOREDIRECTIONBITMAP;
                }
                SetWindowLongPtrW(hwnd, GWL_EXSTYLE, ex);
                if (exstyle == "colorkey") {
                    // Chroma key: DWM keys on RGB, not alpha, so pure black
                    // (the alpha-0 clear after post) becomes see-through
                    // even when the driver presents opaque.
                    SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 255, LWA_COLORKEY);
                } else if (exstyle.find("layered") != std::string::npos) {
                    SetLayeredWindowAttributes(hwnd, 0, 255, LWA_ALPHA);
                }
                log::info("Transparent window ex-style experiment '{}' applied (exstyle 0x{:x})",
                          exstyle, static_cast<unsigned long>(ex));
            }
        }
#endif

        // Center on the requested monitor (SDL3 has no position in CreateWindow).
        int displayCount = 0;
        SDL_DisplayID* displays = SDL_GetDisplays(&displayCount);
        if (displays && desc.monitorIndex >= 0 && desc.monitorIndex < displayCount) {
            const SDL_DisplayID display = displays[desc.monitorIndex];
            SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED_DISPLAY(display),
                                  SDL_WINDOWPOS_CENTERED_DISPLAY(display));
        }
        SDL_free(displays);
        SDL_ShowWindow(window);

        log::info("Created presentation target '{}' ({}x{}, style {})", desc.title, desc.size.width,
                  desc.size.height, static_cast<int>(desc.style));
        return std::unique_ptr<PresentationTarget>(std::make_unique<Sdl3Target>(window, desc.style));
    }

    void setRelativeMouseMode(PresentationTarget& target, bool enabled) override {
        auto& sdlTarget = static_cast<Sdl3Target&>(target);
        SDL_SetWindowRelativeMouseMode(sdlTarget.handle(), enabled);
    }

    std::vector<const char*> requiredVulkanInstanceExtensions() const override {
        Uint32 count = 0;
        const char* const* names = SDL_Vulkan_GetInstanceExtensions(&count);
        return {names, names + count};
    }

    Result<VkSurfaceKHR> createVulkanSurface(VkInstance instance, PresentationTarget& target) override {
        // Safe: this backend only ever hands out Sdl3Target instances.
        auto& sdlTarget = static_cast<Sdl3Target&>(target);
        VkSurfaceKHR surface = nullptr;
        if (!SDL_Vulkan_CreateSurface(sdlTarget.handle(), instance, nullptr, &surface)) {
            return Error{std::string("SDL_Vulkan_CreateSurface failed: ") + SDL_GetError()};
        }
        return surface;
    }

    std::vector<Event> pumpEvents() override {
        std::vector<Event> events;
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
            case SDL_EVENT_QUIT:
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                events.push_back({.type = Event::Type::CloseRequested});
                break;
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                events.push_back({.type = Event::Type::Resized,
                                  .size = {static_cast<std::uint32_t>(e.window.data1),
                                           static_cast<std::uint32_t>(e.window.data2)}});
                break;
            case SDL_EVENT_KEY_DOWN:
                events.push_back({.type = Event::Type::KeyDown, .key = translateKey(e.key.key)});
                break;
            case SDL_EVENT_KEY_UP:
                events.push_back({.type = Event::Type::KeyUp, .key = translateKey(e.key.key)});
                break;
            case SDL_EVENT_MOUSE_MOTION:
                events.push_back({.type = Event::Type::MouseMoved,
                                  .mouseX = e.motion.x,
                                  .mouseY = e.motion.y,
                                  .mouseDeltaX = e.motion.xrel,
                                  .mouseDeltaY = e.motion.yrel});
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
            case SDL_EVENT_MOUSE_BUTTON_UP:
                events.push_back({.type = e.type == SDL_EVENT_MOUSE_BUTTON_DOWN
                                              ? Event::Type::MouseButtonDown
                                              : Event::Type::MouseButtonUp,
                                  .button = translateButton(e.button.button),
                                  .mouseX = e.button.x,
                                  .mouseY = e.button.y});
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                events.push_back({.type = Event::Type::MouseWheel, .wheelDelta = e.wheel.y});
                break;
            default:
                break;
            }
        }
        return events;
    }

private:
    bool initialized_ = false;
};

} // namespace

std::unique_ptr<IPlatformBackend> makeSdl3Backend() { return std::make_unique<Sdl3Backend>(); }

} // namespace rend::platform

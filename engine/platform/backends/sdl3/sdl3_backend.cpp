#include "rend/core/log.h"
#include "rend/platform/backend.h"

#include <SDL3/SDL.h>

namespace rend::platform {

namespace {

Key translateKey(SDL_Keycode code) {
    switch (code) {
    case SDLK_ESCAPE: return Key::Escape;
    case SDLK_SPACE: return Key::Space;
    case SDLK_RETURN: return Key::Enter;
    case SDLK_F11: return Key::F11;
    default: return Key::Unknown;
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
        SDL_WindowFlags flags = 0;
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

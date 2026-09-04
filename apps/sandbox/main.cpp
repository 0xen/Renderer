#include "rend/core/log.h"
#include "rend/platform/backend.h"

using namespace rend;

int main() {
    log::info("Renderer sandbox v0.1.0");

    auto backendResult = platform::createBackend(platform::BackendKind::SDL3);
    if (!backendResult) {
        log::error("Failed to create backend: {}", backendResult.error().message);
        return 1;
    }
    auto backend = std::move(backendResult).value();

    if (auto init = backend->initialize(); !init) {
        log::error("Backend init failed: {}", init.error().message);
        return 1;
    }

    platform::TargetDesc desc{
        .style = platform::WindowStyle::Borderless,
        .size = {1280, 720},
        .title = "Renderer Sandbox",
    };
    auto targetResult = backend->createTarget(desc);
    if (!targetResult) {
        log::error("Failed to create presentation target: {}", targetResult.error().message);
        backend->shutdown();
        return 1;
    }
    auto target = std::move(targetResult).value();

    const auto extent = target->sizeInPixels();
    log::info("Presentation target live at {}x{} pixels — press Esc to quit", extent.width, extent.height);

    bool running = true;
    while (running) {
        for (const auto& event : backend->pumpEvents()) {
            switch (event.type) {
            case platform::Event::Type::CloseRequested:
                running = false;
                break;
            case platform::Event::Type::KeyDown:
                if (event.key == platform::Key::Escape) {
                    running = false;
                }
                break;
            case platform::Event::Type::Resized:
                log::info("Resized to {}x{}", event.size.width, event.size.height);
                break;
            default:
                break;
            }
        }
    }

    log::info("Shutting down");
    target.reset();
    backend->shutdown();
    return 0;
}

#include "rend/core/log.h"
#include "rend/gpu/device.h"
#include "rend/gpu/instance.h"
#include "rend/platform/backend.h"

using namespace rend;

int main(int argc, char** argv) {
    log::info("Renderer viewer v0.1.0");

    // Scene-description XML (shaders, scene setup, models). Parsing lands
    // with the renderer layer; the entry point is established now.
    if (argc > 1) {
        log::info("Scene file requested: {} (XML scene loading not implemented yet)", argv[1]);
    } else {
        log::info("No scene file given (usage: viewer <scene.xml>)");
    }

    auto instanceResult = gpu::Instance::create({.appName = "Renderer Viewer"});
    if (!instanceResult) {
        log::error("Vulkan instance creation failed: {}", instanceResult.error().message);
        return 1;
    }
    auto instance = std::move(instanceResult).value();

    auto deviceResult = gpu::Device::create(*instance, gpu::FeatureSet::gpuDriven());
    if (!deviceResult) {
        log::error("Vulkan device creation failed: {}", deviceResult.error().message);
        return 1;
    }
    auto device = std::move(deviceResult).value();

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
        .title = "Renderer Viewer",
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

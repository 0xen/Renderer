#include "rend/assetio/scene_loader.h"
#include "rend/core/log.h"
#include "rend/core/math.h"
#include "rend/core/paths.h"
#include "rend/gpu/device.h"
#include "rend/gpu/frame_renderer.h"
#include "rend/gpu/instance.h"
#include "rend/gpu/pipeline.h"
#include "rend/gpu/shader.h"
#include "rend/gpu/swapchain.h"
#include "rend/gpu/memory_pool.h"
#include "rend/gpu/transfer.h"
#include "rend/platform/backend.h"

#include <chrono>
#include <optional>

using namespace rend;

namespace {

// Where a mesh landed in the geometry pool; becomes one indirect entry.
struct GeometryLocation {
    gpu::BufferSlice vertices;
    gpu::BufferSlice indices;
    std::uint32_t indexCount = 0;
    std::uint32_t materialIndex = 0;
};

// Interleaved vertex layout of the scene pass: position, normal, uv.
constexpr std::uint32_t kVertexStride = 8 * sizeof(float);

math::Mat4 cameraViewProj(const assetio::CameraDesc& camera, std::uint32_t width,
                          std::uint32_t height) {
    constexpr float kPi = 3.14159265358979323846f;
    const auto& p = camera.position;
    const auto& t = camera.target;
    const math::Mat4 view =
        math::lookAt({p[0], p[1], p[2]}, {t[0], t[1], t[2]}, {0.0f, 1.0f, 0.0f});
    const float aspect = height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
    const math::Mat4 proj =
        math::perspective(camera.fovDegrees * kPi / 180.0f, aspect, 0.1f, 300.0f);
    return math::mul(proj, view);
}

// Interleave the assetio streams into the vertex layout the scene pass will
// consume: position (3f), normal (3f), uv (2f). Missing streams pad with
// zeros so one pipeline serves every mesh.
std::vector<float> interleave(const assetio::MeshData& mesh) {
    const std::size_t count = mesh.vertexCount();
    std::vector<float> out;
    out.reserve(count * 8);
    for (std::size_t i = 0; i < count; ++i) {
        out.insert(out.end(), {mesh.positions[i * 3], mesh.positions[i * 3 + 1],
                               mesh.positions[i * 3 + 2]});
        if (mesh.normals.size() == count * 3) {
            out.insert(out.end(),
                       {mesh.normals[i * 3], mesh.normals[i * 3 + 1], mesh.normals[i * 3 + 2]});
        } else {
            out.insert(out.end(), {0.0f, 0.0f, 0.0f});
        }
        if (mesh.uvs.size() == count * 2) {
            out.insert(out.end(), {mesh.uvs[i * 2], mesh.uvs[i * 2 + 1]});
        } else {
            out.insert(out.end(), {0.0f, 0.0f});
        }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    bool debug = false;
    const char* scenePath = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--debug") {
            debug = true;
        } else {
            scenePath = argv[i];
        }
    }

    // --debug: mirror the log to a file next to the exe (survives crashes,
    // hangs, and closed consoles) and turn on synchronization validation.
    if (debug) {
        log::mirrorToFile(executableDirectory() / "viewer.log");
    }

    log::info("Renderer viewer v0.1.0{}", debug ? " (debug)" : "");

    // The assetio project turns the scene XML + referenced model files into
    // plain CPU-side data; the viewer feeds it to the GPU below.
    std::optional<assetio::LoadedScene> scene;
    if (scenePath) {
        const auto start = std::chrono::steady_clock::now();
        auto sceneResult = assetio::loadScene(scenePath, assetio::ImporterRegistry::withBuiltins());
        if (!sceneResult) {
            log::error("Scene load failed: {}", sceneResult.error().message);
            return 1;
        }
        scene = std::move(sceneResult).value();
        const auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                  start)
                .count();
        std::size_t vertices = 0, triangles = 0;
        for (const auto& model : scene->models) {
            std::size_t modelVerts = 0, modelTris = 0;
            for (const auto& mesh : model.data.meshes) {
                modelVerts += mesh.vertexCount();
                modelTris += mesh.triangleCount();
            }
            log::info("  Model '{}': {} meshes, {} materials, {} vertices, {} triangles",
                      model.desc.name, model.data.meshes.size(), model.data.materials.size(),
                      modelVerts, modelTris);
            vertices += modelVerts;
            triangles += modelTris;
        }
        log::info("Scene '{}' loaded in {} ms: {} models, {} vertices, {} triangles", scene->name,
                  ms, scene->models.size(), vertices, triangles);
    } else {
        log::info("No scene file given (usage: viewer [--debug] <scene.xml>)");
    }

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

    auto instanceResult = gpu::Instance::create({
        .appName = "Renderer Viewer",
        .enableSyncValidation = debug,
        .extraExtensions = backend->requiredVulkanInstanceExtensions(),
    });
    if (!instanceResult) {
        log::error("Vulkan instance creation failed: {}", instanceResult.error().message);
        return 1;
    }
    auto instance = std::move(instanceResult).value();

    auto features = gpu::FeatureSet::gpuDriven();
    features.requiredExtensions.push_back(gpu::kSwapchainExtension);
    auto deviceResult = gpu::Device::create(*instance, features);
    if (!deviceResult) {
        log::error("Vulkan device creation failed: {}", deviceResult.error().message);
        return 1;
    }
    auto device = std::move(deviceResult).value();

    // Milestone 7 step 1: everything the scene pass will draw lives in one
    // device-local memory pool, filled through the transfer queue. Indirect
    // draw entries over these slices come next.
    std::unique_ptr<gpu::MemoryPool> geometryPool;
    std::unique_ptr<gpu::Buffer> indirectBuffer;
    std::vector<GeometryLocation> geometry;
    if (scene) {
        const auto start = std::chrono::steady_clock::now();
        auto poolResult = gpu::MemoryPool::create(*device, 128ull * 1024 * 1024);
        if (!poolResult) {
            log::error("Memory pool creation failed: {}", poolResult.error().message);
            return 1;
        }
        geometryPool = std::move(poolResult).value();

        auto transferResult = gpu::TransferContext::create(*device);
        if (!transferResult) {
            log::error("Transfer context creation failed: {}", transferResult.error().message);
            return 1;
        }
        auto transfer = std::move(transferResult).value();

        for (const auto& model : scene->models) {
            for (const auto& mesh : model.data.meshes) {
                const std::vector<float> vertexData = interleave(mesh);
                // Stride alignment keeps vertexOffset (= offset / stride) exact.
                auto vertexSlice =
                    geometryPool->allocate(vertexData.size() * sizeof(float), kVertexStride);
                auto indexSlice =
                    geometryPool->allocate(mesh.indices.size() * sizeof(std::uint32_t), 4);
                if (!vertexSlice || !indexSlice) {
                    log::error("Memory pool allocation failed: {}",
                               (!vertexSlice ? vertexSlice.error() : indexSlice.error()).message);
                    return 1;
                }
                auto stagedVerts =
                    transfer->stage(geometryPool->buffer(), vertexSlice.value().offset,
                                    vertexData.data(), vertexSlice.value().size);
                auto stagedIndices =
                    transfer->stage(geometryPool->buffer(), indexSlice.value().offset,
                                    mesh.indices.data(), indexSlice.value().size);
                if (!stagedVerts || !stagedIndices) {
                    log::error("Staging failed: {}",
                               (!stagedVerts ? stagedVerts.error() : stagedIndices.error()).message);
                    return 1;
                }
                geometry.push_back({.vertices = vertexSlice.value(),
                                    .indices = indexSlice.value(),
                                    .indexCount = static_cast<std::uint32_t>(mesh.indices.size()),
                                    .materialIndex = mesh.materialIndex});
            }
        }
        // One indirect entry per mesh, addressing its slices by offset.
        // instanceCount is the milestone-7 load/unload toggle; firstInstance
        // becomes the object-SSBO index in step 3.
        std::vector<gpu::DrawIndexedIndirect> draws;
        draws.reserve(geometry.size());
        for (std::size_t i = 0; i < geometry.size(); ++i) {
            const GeometryLocation& location = geometry[i];
            draws.push_back({
                .indexCount = location.indexCount,
                .instanceCount = 1,
                .firstIndex = static_cast<std::uint32_t>(location.indices.offset /
                                                         sizeof(std::uint32_t)),
                .vertexOffset = static_cast<std::int32_t>(location.vertices.offset / kVertexStride),
                .firstInstance = static_cast<std::uint32_t>(i),
            });
        }
        auto indirectResult = gpu::Buffer::create(
            *device, {
                         .size = draws.size() * sizeof(gpu::DrawIndexedIndirect),
                         .usage = gpu::kUsageIndirect | gpu::kUsageTransferDst,
                         .location = gpu::MemoryLocation::DeviceLocal,
                         .sharedWithTransferQueue = true,
                     });
        if (!indirectResult) {
            log::error("Indirect buffer creation failed: {}", indirectResult.error().message);
            return 1;
        }
        indirectBuffer = std::move(indirectResult).value();
        if (auto staged = transfer->stage(*indirectBuffer, 0, draws.data(),
                                          draws.size() * sizeof(gpu::DrawIndexedIndirect));
            !staged) {
            log::error("Indirect staging failed: {}", staged.error().message);
            return 1;
        }

        if (auto flushed = transfer->flush(); !flushed) {
            log::error("Geometry upload failed: {}", flushed.error().message);
            return 1;
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        log::info("Geometry uploaded in {} ms: {} slices, {:.1f} MiB of {} MiB used ({} queue)",
                  ms, geometryPool->allocationCount(),
                  static_cast<double>(geometryPool->usedBytes()) / (1024.0 * 1024.0),
                  geometryPool->capacity() / (1024 * 1024),
                  device->hasDedicatedTransfer() ? "dedicated transfer" : "graphics");
    }

    platform::TargetDesc desc{
        .style = platform::WindowStyle::Decorated,
        .size = {1280, 720},
        .title = "Renderer Viewer",
    };
    auto targetResult = backend->createTarget(desc);
    if (!targetResult) {
        log::error("Failed to create presentation target: {}", targetResult.error().message);
        return 1;
    }
    auto target = std::move(targetResult).value();

    auto surfaceResult = backend->createVulkanSurface(instance->handle(), *target);
    if (!surfaceResult) {
        log::error("Surface creation failed: {}", surfaceResult.error().message);
        return 1;
    }

    const auto extent = target->sizeInPixels();
    auto swapchainResult = gpu::Swapchain::create(*instance, *device,
                                                  {
                                                      .surface = surfaceResult.value(),
                                                      .width = extent.width,
                                                      .height = extent.height,
                                                      .transparent = false,
                                                      .vsync = true,
                                                  });
    if (!swapchainResult) {
        log::error("Swapchain creation failed: {}", swapchainResult.error().message);
        return 1;
    }
    auto swapchain = std::move(swapchainResult).value();

    // Shaders are compiled offline (dxc) into data/shaders next to the exe.
    const auto shaderDir = executableDirectory() / "data" / "shaders";
    auto vertexResult = gpu::Shader::createFromFile(*device, shaderDir / "triangle.vert.spv");
    if (!vertexResult) {
        log::error("{}", vertexResult.error().message);
        return 1;
    }
    auto fragmentResult = gpu::Shader::createFromFile(*device, shaderDir / "triangle.frag.spv");
    if (!fragmentResult) {
        log::error("{}", fragmentResult.error().message);
        return 1;
    }
    auto vertexShader = std::move(vertexResult).value();
    auto fragmentShader = std::move(fragmentResult).value();

    auto pipelineResult = gpu::Pipeline::createGraphics(*device,
                                                        {
                                                            .vertexShader = vertexShader.get(),
                                                            .fragmentShader = fragmentShader.get(),
                                                            .colorFormat = swapchain->imageFormat(),
                                                        });
    if (!pipelineResult) {
        log::error("Pipeline creation failed: {}", pipelineResult.error().message);
        return 1;
    }
    auto pipeline = std::move(pipelineResult).value();

    // Scene pass pipeline: interleaved vertex input from the geometry pool,
    // depth-tested, camera via push constant.
    std::unique_ptr<gpu::Pipeline> scenePipeline;
    std::unique_ptr<gpu::Shader> sceneVert, sceneFrag;
    if (scene) {
        auto vertResult = gpu::Shader::createFromFile(*device, shaderDir / "scene.vert.spv");
        auto fragResult = gpu::Shader::createFromFile(*device, shaderDir / "scene.frag.spv");
        if (!vertResult || !fragResult) {
            log::error("{}", (!vertResult ? vertResult : fragResult).error().message);
            return 1;
        }
        sceneVert = std::move(vertResult).value();
        sceneFrag = std::move(fragResult).value();
        auto sceneResult = gpu::Pipeline::createGraphics(
            *device, {
                         .vertexShader = sceneVert.get(),
                         .fragmentShader = sceneFrag.get(),
                         .colorFormat = swapchain->imageFormat(),
                         .vertexStride = kVertexStride,
                         .vertexAttributes = {{0, gpu::kFormatR32G32B32Sfloat, 0},
                                              {1, gpu::kFormatR32G32B32Sfloat, 12},
                                              {2, gpu::kFormatR32G32Sfloat, 24}},
                         .depthFormat = gpu::kFormatD32Sfloat,
                         .pushConstantBytes = 64,
                     });
        if (!sceneResult) {
            log::error("Scene pipeline creation failed: {}", sceneResult.error().message);
            return 1;
        }
        scenePipeline = std::move(sceneResult).value();
    }

    auto rendererResult = gpu::FrameRenderer::create(*device, *swapchain);
    if (!rendererResult) {
        log::error("Frame renderer creation failed: {}", rendererResult.error().message);
        return 1;
    }
    auto renderer = std::move(rendererResult).value();

    // With a scene, every frame is the indirect batch over the geometry
    // pool; without one, the milestone-6 triangle stays as the fallback.
    const bool drawScene = scenePipeline && indirectBuffer && !geometry.empty();
    gpu::DrawBatch batch;
    if (drawScene) {
        batch.geometry = geometryPool->buffer().handle();
        batch.indirect = indirectBuffer->handle();
        batch.drawCount = static_cast<std::uint32_t>(geometry.size());
        batch.viewProj = cameraViewProj(scene->camera, extent.width, extent.height);
        log::info("Scene pass ready: {} indirect draws", batch.drawCount);
    }

    log::info("Viewer live at {}x{} — press Esc to quit", extent.width, extent.height);

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
                renderer->resize(event.size.width, event.size.height);
                if (drawScene && event.size.width > 0 && event.size.height > 0) {
                    batch.viewProj =
                        cameraViewProj(scene->camera, event.size.width, event.size.height);
                }
                break;
            default:
                break;
            }
        }
        if (!running) {
            break;
        }
        if (auto r = renderer->drawFrame(drawScene ? *scenePipeline : *pipeline,
                                         drawScene ? &batch : nullptr);
            !r) {
            log::error("Frame failed: {}", r.error().message);
            running = false;
        }
    }

    log::info("Shutting down");
    renderer->waitIdle();
    // The swapchain goes first: destroying it retires presents that are still
    // waiting on the frame renderer's per-image semaphores, which the renderer
    // then destroys. It also owns the surface, so it must precede the instance.
    swapchain.reset();
    renderer.reset();
    pipeline.reset();
    fragmentShader.reset();
    vertexShader.reset();
    target.reset();
    backend->shutdown();
    return 0;
}

#include "rend/assetio/scene_loader.h"
#include "rend/assetio/texture_loader.h"
#include "rend/core/log.h"
#include "rend/core/math.h"
#include "rend/core/paths.h"
#include "rend/gpu/descriptor_table.h"
#include "rend/gpu/texture_uploader.h"
#include "rend/gpu/device.h"
#include "rend/gpu/frame_renderer.h"
#include "rend/gpu/instance.h"
#include "rend/gpu/pipeline.h"
#include "rend/gpu/shader.h"
#include "rend/gpu/swapchain.h"
#include "rend/gpu/memory_pool.h"
#include "rend/gpu/transfer.h"
#include "rend/platform/backend.h"

#include "ui.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <format>
#include <cstddef>
#include <cstring>
#include <optional>
#include <unordered_map>

using namespace rend;

namespace {

// Where a mesh landed in the geometry pool; becomes one indirect entry.
struct GeometryLocation {
    gpu::BufferSlice vertices;
    gpu::BufferSlice indices;
    std::uint32_t indexCount = 0;
    std::uint32_t materialIndex = 0;
};

// One row per object in the bindless table's SSBO, found by the indirect
// entry's firstInstance. Must match ObjectData in scene.hlsl.
struct ObjectData {
    std::uint32_t textureIndex = 0;
    std::uint32_t alphaMasked = 0;
    float alphaCutoff = 0.5f;
    float pad = 0.0f;
};

// Interleaved vertex layout of the scene pass: position, normal, uv.
constexpr std::uint32_t kVertexStride = 8 * sizeof(float);

constexpr float kPi = 3.14159265358979323846f;

// Free-fly camera: yaw/pitch angles plus position, driven by RMB mouselook
// and WASD/QE. Yaw 0 looks down -Z (matching math::lookAt's convention).
struct FlyCamera {
    math::Vec3 position{};
    float yaw = 0.0f;   // radians, positive turns right (+X)
    float pitch = 0.0f; // radians, positive looks up; clamped near +/-90
    float fovDegrees = 60.0f;

    math::Vec3 forward() const {
        const float cp = std::cos(pitch);
        return {cp * std::sin(yaw), std::sin(pitch), -cp * std::cos(yaw)};
    }

    math::Mat4 viewProj(std::uint32_t width, std::uint32_t height) const {
        const math::Vec3 f = forward();
        const math::Mat4 view = math::lookAt(
            position, {position.x + f.x, position.y + f.y, position.z + f.z}, {0.0f, 1.0f, 0.0f});
        const float aspect =
            height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
        return math::mul(math::perspective(fovDegrees * kPi / 180.0f, aspect, 0.1f, 300.0f), view);
    }

    static FlyCamera fromScene(const assetio::CameraDesc& camera) {
        FlyCamera out;
        out.position = {camera.position[0], camera.position[1], camera.position[2]};
        const math::Vec3 d = math::normalize(math::sub(
            {camera.target[0], camera.target[1], camera.target[2]}, out.position));
        out.yaw = std::atan2(d.x, -d.z);
        out.pitch = std::asin(std::clamp(d.y, -1.0f, 1.0f));
        out.fovDegrees = camera.fovDegrees;
        return out;
    }
};

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
    bool vsync = true;
    bool staticMode = false;
    std::uint64_t benchFrames = 0; // non-zero: exit after N frames with a report
    // Caps the draw-submit ladder for testing the fallbacks; the actual mode
    // is still limited by what the device supports.
    auto maxDrawMode = gpu::DrawSubmitMode::IndirectCount;
    const char* scenePath = nullptr;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--debug") {
            debug = true;
        } else if (arg == "--novsync") {
            vsync = false;
        } else if (arg == "--static") {
            staticMode = true;
        } else if (arg == "--bench" && i + 1 < argc) {
            const std::string_view count = argv[++i];
            std::from_chars(count.data(), count.data() + count.size(), benchFrames);
        } else if (arg == "--draw-mode" && i + 1 < argc) {
            const std::string_view mode = argv[++i];
            if (mode == "count") {
                maxDrawMode = gpu::DrawSubmitMode::IndirectCount;
            } else if (mode == "indirect") {
                maxDrawMode = gpu::DrawSubmitMode::Indirect;
            } else if (mode == "direct") {
                maxDrawMode = gpu::DrawSubmitMode::Direct;
            } else {
                log::error("Unknown --draw-mode '{}' (count|indirect|direct)", mode);
                return 1;
            }
        } else {
            scenePath = arg.data();
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
        log::info("No scene file given "
                  "(usage: viewer [--debug] [--novsync] [--static] [--bench N] "
                  "[--draw-mode count|indirect|direct] <scene.xml>)");
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
    std::unique_ptr<gpu::Buffer> compactedBuffer;
    std::unique_ptr<gpu::Buffer> countBuffer;
    std::unique_ptr<gpu::Buffer> cameraBuffer; // one viewProj per frame slot, CPU-written
    std::vector<gpu::DrawIndexedIndirect> draws; // outlives the loop: Direct mode records from it
    std::unique_ptr<gpu::Buffer> objectBuffer;
    std::unique_ptr<gpu::DescriptorTable> descriptorTable;
    std::vector<std::unique_ptr<gpu::Image>> textures;
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

        // Slot 0 of the bindless texture array is a 1x1 white fallback so
        // untextured materials sample neutrally.
        std::vector<std::filesystem::path> texturePaths{{}};
        std::unordered_map<std::string, std::uint32_t> textureSlotByPath;
        std::vector<ObjectData> objectData;

        for (const auto& model : scene->models) {
            for (const auto& mesh : model.data.meshes) {
                ObjectData object;
                if (mesh.materialIndex < model.data.materials.size()) {
                    const auto& material = model.data.materials[mesh.materialIndex];
                    object.alphaMasked = material.alphaMasked ? 1u : 0u;
                    object.alphaCutoff = material.alphaCutoff;
                    if (!material.baseColorTexture.empty()) {
                        const std::string key = material.baseColorTexture.string();
                        auto [it, inserted] = textureSlotByPath.try_emplace(
                            key, static_cast<std::uint32_t>(texturePaths.size()));
                        if (inserted) {
                            texturePaths.push_back(material.baseColorTexture);
                        }
                        object.textureIndex = it->second;
                    }
                }
                objectData.push_back(object);

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
        // Host-visible with one region per frame in flight: the CPU rewrites
        // the current slot's instanceCounts every frame (the milestone-7
        // load/unload toggle) while the other slot's region is in flight.
        const std::uint64_t indirectRegion = draws.size() * sizeof(gpu::DrawIndexedIndirect);
        auto indirectResult = gpu::Buffer::create(
            *device, {
                         .size = indirectRegion * gpu::FrameRenderer::kFramesInFlight,
                         // Storage too: the cull pass reads these entries as
                         // its draw templates (descriptor binding 3).
                         .usage = gpu::kUsageIndirect | gpu::kUsageStorage,
                         .location = gpu::MemoryLocation::HostVisible,
                     });
        if (!indirectResult) {
            log::error("Indirect buffer creation failed: {}", indirectResult.error().message);
            return 1;
        }
        indirectBuffer = std::move(indirectResult).value();
        for (std::uint32_t slot = 0; slot < gpu::FrameRenderer::kFramesInFlight; ++slot) {
            std::memcpy(static_cast<std::byte*>(indirectBuffer->mapped()) + slot * indirectRegion,
                        draws.data(), indirectRegion);
        }

        // Cull-pass output: the compacted indirect list the GPU builds each
        // frame (per-slot regions like the templates).
        auto compactedResult = gpu::Buffer::create(
            *device, {
                         .size = indirectRegion * gpu::FrameRenderer::kFramesInFlight,
                         .usage = gpu::kUsageIndirect | gpu::kUsageStorage,
                         .location = gpu::MemoryLocation::DeviceLocal,
                     });
        if (!compactedResult) {
            log::error("Compacted buffer creation failed: {}", compactedResult.error().message);
            return 1;
        }
        compactedBuffer = std::move(compactedResult).value();

        // Draw-count buffer for IndirectCount mode: one uint32 per frame
        // slot, written by the cull pass (zeroed via fill, incremented by
        // the shader) and read by vkCmdDrawIndexedIndirectCount.
        auto countResult = gpu::Buffer::create(
            *device, {
                         .size = sizeof(std::uint32_t) * gpu::FrameRenderer::kFramesInFlight,
                         .usage = gpu::kUsageIndirect | gpu::kUsageStorage |
                                  gpu::kUsageTransferDst,
                         .location = gpu::MemoryLocation::DeviceLocal,
                     });
        if (!countResult) {
            log::error("Count buffer creation failed: {}", countResult.error().message);
            return 1;
        }
        countBuffer = std::move(countResult).value();

        // Camera matrices, one region per frame slot: rewritten by the CPU
        // every frame (after waitFrameSlot), read by the vertex shader via
        // the slot index push constant. This is what keeps static command
        // buffers valid while the camera moves.
        auto cameraResult = gpu::Buffer::create(
            *device, {
                         .size = sizeof(math::Mat4) * gpu::FrameRenderer::kFramesInFlight,
                         .usage = gpu::kUsageStorage,
                         .location = gpu::MemoryLocation::HostVisible,
                     });
        if (!cameraResult) {
            log::error("Camera buffer creation failed: {}", cameraResult.error().message);
            return 1;
        }
        cameraBuffer = std::move(cameraResult).value();

        auto objectResult = gpu::Buffer::create(
            *device, {
                         .size = objectData.size() * sizeof(ObjectData),
                         .usage = gpu::kUsageStorage | gpu::kUsageTransferDst,
                         .location = gpu::MemoryLocation::DeviceLocal,
                         .sharedWithTransferQueue = true,
                     });
        if (!objectResult) {
            log::error("Object buffer creation failed: {}", objectResult.error().message);
            return 1;
        }
        objectBuffer = std::move(objectResult).value();
        if (auto staged = transfer->stage(*objectBuffer, 0, objectData.data(),
                                          objectData.size() * sizeof(ObjectData));
            !staged) {
            log::error("Object staging failed: {}", staged.error().message);
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

        // Bindless table: decode every referenced base-color texture through
        // assetio, upload with full mip chains, and point the object SSBO
        // rows at their slots.
        const auto texStart = std::chrono::steady_clock::now();
        auto tableResult = gpu::DescriptorTable::create(*device, 1024);
        if (!tableResult) {
            log::error("Descriptor table creation failed: {}", tableResult.error().message);
            return 1;
        }
        descriptorTable = std::move(tableResult).value();
        descriptorTable->writeObjectBuffer(objectBuffer->handle(),
                                           objectData.size() * sizeof(ObjectData));
        descriptorTable->writeStorageBuffer(3, indirectBuffer->handle(), indirectBuffer->size());
        descriptorTable->writeStorageBuffer(4, compactedBuffer->handle(),
                                            compactedBuffer->size());
        descriptorTable->writeStorageBuffer(5, countBuffer->handle(), countBuffer->size());
        descriptorTable->writeStorageBuffer(6, cameraBuffer->handle(), cameraBuffer->size());

        auto uploaderResult = gpu::TextureUploader::create(*device);
        if (!uploaderResult) {
            log::error("Texture uploader creation failed: {}", uploaderResult.error().message);
            return 1;
        }
        auto uploader = std::move(uploaderResult).value();

        std::uint64_t texelBytes = 0;
        for (std::size_t i = 0; i < texturePaths.size(); ++i) {
            Result<std::unique_ptr<gpu::Image>> uploaded = [&]() {
                if (i == 0) {
                    const std::uint8_t white[4] = {255, 255, 255, 255};
                    return uploader->upload(1, 1, white);
                }
                auto decoded = assetio::loadTexture(texturePaths[i]);
                if (!decoded) {
                    return Result<std::unique_ptr<gpu::Image>>{decoded.error()};
                }
                const auto& t = decoded.value();
                texelBytes += t.rgba.size();
                return uploader->upload(t.width, t.height, t.rgba.data());
            }();
            if (!uploaded) {
                log::error("Texture {} failed: {}", texturePaths[i].filename().string(),
                           uploaded.error().message);
                return 1;
            }
            descriptorTable->writeTexture(static_cast<std::uint32_t>(i),
                                          uploaded.value()->view());
            textures.push_back(std::move(uploaded).value());
        }
        const auto texMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - texStart)
                               .count();
        log::info("Textures ready in {} ms: {} images ({:.1f} MiB decoded), mipmapped, bindless",
                  texMs, textures.size(), static_cast<double>(texelBytes) / (1024.0 * 1024.0));
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
                                                      .vsync = vsync,
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
    std::unique_ptr<gpu::Pipeline> cullPipeline;
    std::unique_ptr<gpu::Shader> sceneVert, sceneFrag, cullShader;
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
                         .pushConstantBytes = sizeof(std::uint32_t), // camera slot index
                         .descriptorLayout = descriptorTable->layout(),
                     });
        if (!sceneResult) {
            log::error("Scene pipeline creation failed: {}", sceneResult.error().message);
            return 1;
        }
        scenePipeline = std::move(sceneResult).value();

        // The compaction pass (IndirectCount mode only). A failure here is
        // not fatal: the draw-mode ladder just skips to Indirect.
        auto cullShaderResult = gpu::Shader::createFromFile(*device, shaderDir / "cull.comp.spv");
        if (cullShaderResult) {
            cullShader = std::move(cullShaderResult).value();
            auto cullResult = gpu::Pipeline::createCompute(
                *device, {
                             .shader = cullShader.get(),
                             .descriptorLayout = descriptorTable->layout(),
                             .pushConstantBytes = 2 * sizeof(std::uint32_t),
                         });
            if (cullResult) {
                cullPipeline = std::move(cullResult).value();
            } else {
                log::warn("Cull pipeline unavailable: {}", cullResult.error().message);
            }
        } else {
            log::warn("Cull shader unavailable: {}", cullShaderResult.error().message);
        }
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
        // Pick the best submit mode the device's enabled features allow,
        // never exceeding the --draw-mode cap. Each tier falls back to the
        // next; Direct works everywhere.
        const bool canIndirect = device->isEnabled(gpu::Feature::MultiDrawIndirect) &&
                                 device->isEnabled(gpu::Feature::DrawIndirectFirstInstance);
        // The count tier now IS the compaction pass, so it also needs the
        // cull pipeline to have built.
        const bool canCount = canIndirect && device->isEnabled(gpu::Feature::DrawIndirectCount) &&
                              cullPipeline != nullptr;
        batch.mode = gpu::DrawSubmitMode::Direct;
        if (canIndirect && maxDrawMode != gpu::DrawSubmitMode::Direct) {
            batch.mode = gpu::DrawSubmitMode::Indirect;
        }
        if (canCount && maxDrawMode == gpu::DrawSubmitMode::IndirectCount) {
            batch.mode = gpu::DrawSubmitMode::IndirectCount;
        }

        batch.geometry = geometryPool->buffer().handle();
        batch.drawCount = static_cast<std::uint32_t>(geometry.size());
        batch.indirectRegionStride = geometry.size() * sizeof(gpu::DrawIndexedIndirect);
        if (batch.mode == gpu::DrawSubmitMode::IndirectCount) {
            // The GPU draws what the cull pass compacted, not the templates.
            batch.indirect = compactedBuffer->handle();
            batch.cullPipeline = cullPipeline.get();
        } else {
            batch.indirect = indirectBuffer->handle();
        }
        batch.count = countBuffer->handle();
        batch.countRegionStride = sizeof(std::uint32_t);
        batch.cpuDraws = draws.data();
        batch.descriptors = descriptorTable->set();

        const char* modeName = batch.mode == gpu::DrawSubmitMode::IndirectCount
                                   ? "indirect-count + GPU compaction"
                               : batch.mode == gpu::DrawSubmitMode::Indirect ? "indirect"
                                                                             : "direct";
        log::info("Scene pass ready: {} draws, {} mode (device: indirect {}, indirect-count {})",
                  batch.drawCount, modeName, canIndirect ? "yes" : "NO",
                  canCount ? "yes" : "NO");
    }

    // Debug UI (ImGui): drawn through the frame renderer's overlay pass,
    // which stays per-frame even when the scene buffers are static.
    auto ui = viewer::Ui::create(*instance, *device, *swapchain);
    if (ui) {
        renderer->setOverlayRecorder([&ui](VkCommandBuffer cmd) { ui->render(cmd); });
    } else {
        log::warn("Debug UI unavailable; continuing without it");
    }

    renderer->setStaticRecording(staticMode);
    log::info("Viewer live at {}x{} — {} recording, vsync {} — Esc quits, Space toggles mode",
              extent.width, extent.height, staticMode ? "static" : "per-frame", vsync ? "on" : "off");

    // Stats window: wall time + renderer CPU counters, reported per mode.
    constexpr std::uint64_t kReportInterval = 600;
    auto reportStart = std::chrono::steady_clock::now();
    auto report = [&](std::uint64_t windowFrames) {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - reportStart).count();
        reportStart = now;
        const gpu::FrameRenderer::Stats stats = renderer->takeStats();
        if (windowFrames == 0 || seconds <= 0.0) {
            return;
        }
        log::info("[{}] {} frames in {:.2f} s = {:.0f} fps | record {:.1f} us/frame{}",
                  renderer->staticRecording() ? "static" : "rerecord", windowFrames, seconds,
                  windowFrames / seconds,
                  stats.frames > 0 ? static_cast<double>(stats.recordMicros) / stats.frames : 0.0,
                  stats.prerecords > 0 ? std::format(" | {} prerecords", stats.prerecords) : "");
    };

    FlyCamera camera;
    if (scene) {
        camera = FlyCamera::fromScene(scene->camera);
    }
    // Held-key state for camera movement; mouselook while RMB is held.
    bool keyHeld[static_cast<int>(platform::Key::LeftCtrl) + 1] = {};
    bool mouselook = false;
    constexpr float kLookSensitivity = 0.0025f; // radians per pixel
    constexpr float kMoveSpeed = 3.0f;          // units per second
    constexpr float kFastMultiplier = 5.0f;

    bool running = true;
    std::uint64_t frame = 0;
    std::uint32_t viewWidth = extent.width;
    std::uint32_t viewHeight = extent.height;
    auto lastFrameTime = std::chrono::steady_clock::now();
    while (running) {
        for (const auto& event : backend->pumpEvents()) {
            if (ui) {
                ui->handleEvent(event);
            }
            switch (event.type) {
            case platform::Event::Type::CloseRequested:
                running = false;
                break;
            case platform::Event::Type::KeyDown:
                keyHeld[static_cast<int>(event.key)] = true;
                if (event.key == platform::Key::Escape) {
                    running = false;
                }
                if (event.key == platform::Key::Space) {
                    report(frame % kReportInterval);
                    renderer->setStaticRecording(!renderer->staticRecording());
                    log::info("Switched to {} recording",
                              renderer->staticRecording() ? "static" : "per-frame");
                }
                break;
            case platform::Event::Type::KeyUp:
                keyHeld[static_cast<int>(event.key)] = false;
                break;
            case platform::Event::Type::MouseButtonDown:
                if (event.button == platform::MouseButton::Right) {
                    mouselook = true;
                    backend->setRelativeMouseMode(*target, true);
                }
                break;
            case platform::Event::Type::MouseButtonUp:
                if (event.button == platform::MouseButton::Right) {
                    mouselook = false;
                    backend->setRelativeMouseMode(*target, false);
                }
                break;
            case platform::Event::Type::MouseMoved:
                if (mouselook) {
                    camera.yaw += event.mouseDeltaX * kLookSensitivity;
                    camera.pitch = std::clamp(camera.pitch - event.mouseDeltaY * kLookSensitivity,
                                              -0.49f * kPi, 0.49f * kPi);
                }
                break;
            case platform::Event::Type::Resized:
                renderer->resize(event.size.width, event.size.height);
                viewWidth = event.size.width;
                viewHeight = event.size.height;
                break;
            default:
                break;
            }
        }
        if (!running) {
            break;
        }

        const auto now = std::chrono::steady_clock::now();
        const float deltaSeconds = std::chrono::duration<float>(now - lastFrameTime).count();
        lastFrameTime = now;

        // Camera movement from held keys (fly style: WASD in the view
        // plane, Q/E down/up, Shift fast).
        if (drawScene) {
            auto held = [&](platform::Key k) { return keyHeld[static_cast<int>(k)]; };
            const float speed = kMoveSpeed *
                                (held(platform::Key::LeftShift) ? kFastMultiplier : 1.0f) *
                                deltaSeconds;
            const math::Vec3 f = camera.forward();
            const math::Vec3 right = math::normalize(math::cross(f, {0.0f, 1.0f, 0.0f}));
            auto move = [&](const math::Vec3& d, float s) {
                camera.position = {camera.position.x + d.x * s, camera.position.y + d.y * s,
                                   camera.position.z + d.z * s};
            };
            if (held(platform::Key::W)) move(f, speed);
            if (held(platform::Key::S)) move(f, -speed);
            if (held(platform::Key::D)) move(right, speed);
            if (held(platform::Key::A)) move(right, -speed);
            if (held(platform::Key::E)) move({0.0f, 1.0f, 0.0f}, speed);
            if (held(platform::Key::Q)) move({0.0f, 1.0f, 0.0f}, -speed);

            // Publish this frame's camera into the slot's region: safe to
            // write once the slot's previous submission retired.
            if (auto r = renderer->waitFrameSlot(); !r) {
                log::error("Frame failed: {}", r.error().message);
                break;
            }
            const math::Mat4 viewProj = camera.viewProj(viewWidth, viewHeight);
            std::memcpy(static_cast<std::byte*>(cameraBuffer->mapped()) +
                            renderer->frameSlot() * sizeof(math::Mat4),
                        viewProj.data(), sizeof(math::Mat4));
        }

        if (ui && viewWidth > 0 && viewHeight > 0) {
            ui->buildFrame(viewWidth, viewHeight, deltaSeconds);
        }

        if (auto r = renderer->drawFrame(drawScene ? *scenePipeline : *pipeline,
                                         drawScene ? &batch : nullptr);
            !r) {
            log::error("Frame failed: {}", r.error().message);
            running = false;
        }

        ++frame;
        if (frame % kReportInterval == 0) {
            report(kReportInterval);
        }
        if (benchFrames != 0 && frame >= benchFrames) {
            report(frame % kReportInterval);
            log::info("Benchmark complete after {} frames", frame);
            running = false;
        }
    }

    log::info("Shutting down");
    renderer->waitIdle();
    ui.reset(); // ImGui's Vulkan objects go while the device is idle and alive
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

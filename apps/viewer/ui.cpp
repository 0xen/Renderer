#include "ui.h"

#include "rend/core/log.h"
#include "rend/gpu/device.h"
#include "rend/gpu/instance.h"
#include "rend/gpu/memory_tracker.h"
#include "rend/gpu/swapchain.h"
#include "rend/platform/events.h"

#include <volk.h>

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>

namespace viewer {

namespace {

// Black and red theme: near-black surfaces, red accents, warm off-white
// text.
void applyTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding = 3.0f;

    const ImVec4 black{0.06f, 0.02f, 0.02f, 0.96f};
    const ImVec4 blackDeep{0.03f, 0.01f, 0.01f, 1.00f};
    const ImVec4 redDim{0.35f, 0.06f, 0.06f, 1.00f};
    const ImVec4 red{0.60f, 0.08f, 0.08f, 1.00f};
    const ImVec4 redHot{0.85f, 0.12f, 0.12f, 1.00f};
    const ImVec4 text{0.94f, 0.88f, 0.88f, 1.00f};

    ImVec4* c = style.Colors;
    c[ImGuiCol_Text] = text;
    c[ImGuiCol_TextDisabled] = ImVec4{0.55f, 0.42f, 0.42f, 1.0f};
    c[ImGuiCol_WindowBg] = black;
    c[ImGuiCol_ChildBg] = ImVec4{0.0f, 0.0f, 0.0f, 0.0f};
    c[ImGuiCol_PopupBg] = blackDeep;
    c[ImGuiCol_Border] = redDim;
    c[ImGuiCol_FrameBg] = ImVec4{0.14f, 0.04f, 0.04f, 1.0f};
    c[ImGuiCol_FrameBgHovered] = redDim;
    c[ImGuiCol_FrameBgActive] = red;
    c[ImGuiCol_TitleBg] = blackDeep;
    c[ImGuiCol_TitleBgActive] = red;
    c[ImGuiCol_TitleBgCollapsed] = blackDeep;
    c[ImGuiCol_MenuBarBg] = blackDeep;
    c[ImGuiCol_ScrollbarBg] = blackDeep;
    c[ImGuiCol_ScrollbarGrab] = redDim;
    c[ImGuiCol_ScrollbarGrabHovered] = red;
    c[ImGuiCol_ScrollbarGrabActive] = redHot;
    c[ImGuiCol_CheckMark] = redHot;
    c[ImGuiCol_SliderGrab] = red;
    c[ImGuiCol_SliderGrabActive] = redHot;
    c[ImGuiCol_Button] = redDim;
    c[ImGuiCol_ButtonHovered] = red;
    c[ImGuiCol_ButtonActive] = redHot;
    c[ImGuiCol_Header] = redDim;
    c[ImGuiCol_HeaderHovered] = red;
    c[ImGuiCol_HeaderActive] = redHot;
    c[ImGuiCol_Separator] = redDim;
    c[ImGuiCol_SeparatorHovered] = red;
    c[ImGuiCol_SeparatorActive] = redHot;
    c[ImGuiCol_ResizeGrip] = redDim;
    c[ImGuiCol_ResizeGripHovered] = red;
    c[ImGuiCol_ResizeGripActive] = redHot;
    c[ImGuiCol_Tab] = ImVec4{0.20f, 0.04f, 0.04f, 1.0f};
    c[ImGuiCol_TabHovered] = red;
    c[ImGuiCol_TabSelected] = red;
    c[ImGuiCol_PlotLines] = redHot;
    c[ImGuiCol_PlotHistogram] = redHot;
}

} // namespace

std::unique_ptr<Ui> Ui::create(const rend::gpu::Instance& instance,
                               const rend::gpu::Device& device,
                               const rend::gpu::Swapchain& swapchain) {
    // The viewer's own volk table (the engine DLL has a separate one); the
    // ImGui Vulkan backend is compiled against it.
    if (volkInitialize() != VK_SUCCESS) {
        rend::log::error("UI: volkInitialize failed");
        return nullptr;
    }
    volkLoadInstance(instance.handle());

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr; // no imgui.ini next to the exe
    applyTheme();

    const VkFormat colorFormat = static_cast<VkFormat>(swapchain.imageFormat());
    ImGui_ImplVulkan_InitInfo info{};
    info.Instance = instance.handle();
    info.PhysicalDevice = device.physicalDevice();
    info.Device = device.handle();
    info.QueueFamily = device.graphicsQueue().familyIndex;
    info.Queue = device.graphicsQueue().queue;
    info.DescriptorPoolSize = 8; // backend creates its own pool
    info.MinImageCount = 2;
    info.ImageCount = static_cast<std::uint32_t>(swapchain.images().size());
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.UseDynamicRendering = true;
    info.PipelineRenderingCreateInfo = {};
    info.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &colorFormat;
    if (!ImGui_ImplVulkan_Init(&info)) {
        rend::log::error("UI: ImGui Vulkan backend init failed");
        ImGui::DestroyContext();
        return nullptr;
    }

    rend::log::info("UI ready (Dear ImGui {}, black/red theme)", IMGUI_VERSION);
    return std::unique_ptr<Ui>(new Ui());
}

Ui::~Ui() {
    ImGui_ImplVulkan_Shutdown();
    ImGui::DestroyContext();
}

void Ui::handleEvent(const rend::platform::Event& event) {
    using rend::platform::Event;
    using rend::platform::MouseButton;
    ImGuiIO& io = ImGui::GetIO();
    switch (event.type) {
    case Event::Type::MouseMoved:
        io.AddMousePosEvent(event.mouseX, event.mouseY);
        break;
    case Event::Type::MouseButtonDown:
    case Event::Type::MouseButtonUp: {
        const int button = event.button == MouseButton::Left    ? ImGuiMouseButton_Left
                           : event.button == MouseButton::Right ? ImGuiMouseButton_Right
                                                                : ImGuiMouseButton_Middle;
        io.AddMouseButtonEvent(button, event.type == Event::Type::MouseButtonDown);
        break;
    }
    case Event::Type::MouseWheel:
        io.AddMouseWheelEvent(0.0f, event.wheelDelta);
        break;
    default:
        break;
    }
}

void Ui::buildFrame(std::uint32_t width, std::uint32_t height, float deltaSeconds,
                    bool* vsync, const LoadingStatus* loading, const CullStats* cull) {
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(width), static_cast<float>(height));
    io.DeltaTime = deltaSeconds > 0.0f ? deltaSeconds : 1.0f / 60.0f;

    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();

    // Debug panel, top-left corner: smoothed FPS (readable instead of
    // flickering), a raw per-frame FPS history graph, and the VSync toggle.
    smoothedFrameSeconds_ = smoothedFrameSeconds_ <= 0.0f
                                ? io.DeltaTime
                                : smoothedFrameSeconds_ * 0.95f + io.DeltaTime * 0.05f;
    fpsHistory_[fpsHistoryOffset_] = 1.0f / io.DeltaTime;
    fpsHistoryOffset_ = (fpsHistoryOffset_ + 1) % fpsHistory_.size();
    ImGui::SetNextWindowPos(ImVec2(8.0f, 8.0f));
    ImGui::Begin("##debug", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
    ImGui::Text("%.0f FPS (%.2f ms)", 1.0f / smoothedFrameSeconds_,
                smoothedFrameSeconds_ * 1000.0f);
    ImGui::PlotLines("##fpsHistory", fpsHistory_.data(),
                     static_cast<int>(fpsHistory_.size()),
                     static_cast<int>(fpsHistoryOffset_), nullptr, 0.0f, FLT_MAX,
                     ImVec2(180.0f, 42.0f));
    if (vsync != nullptr) {
        ImGui::Checkbox("VSync", vsync);
    }

    // GPU memory ledger, same treatment as FPS: a live total, a history
    // graph, and the per-kind split. Every gpu::Buffer/Image reports its
    // device allocation on construction and destruction, so this is the
    // whole device-memory truth (minus the swapchain, which the driver
    // owns).
    const auto mem = rend::gpu::MemoryTracker::snapshot();
    constexpr double kMiB = 1024.0 * 1024.0;
    const double totalMiB = static_cast<double>(mem.totalBytes()) / kMiB;
    memoryHistory_[memoryHistoryOffset_] = static_cast<float>(totalMiB);
    memoryHistoryOffset_ = (memoryHistoryOffset_ + 1) % memoryHistory_.size();
    ImGui::Separator();
    ImGui::Text("GPU memory: %.1f MiB (%u allocations)", totalMiB, mem.totalCount());
    ImGui::PlotLines("##memHistory", memoryHistory_.data(),
                     static_cast<int>(memoryHistory_.size()),
                     static_cast<int>(memoryHistoryOffset_), nullptr, 0.0f, FLT_MAX,
                     ImVec2(180.0f, 42.0f));
    using Kind = rend::gpu::MemoryTracker::Kind;
    ImGui::Text("Buffers: %.1f MiB (%u) device, %.1f MiB (%u) host",
                mem.bytes[static_cast<std::uint32_t>(Kind::DeviceBuffer)] / kMiB,
                mem.counts[static_cast<std::uint32_t>(Kind::DeviceBuffer)],
                mem.bytes[static_cast<std::uint32_t>(Kind::HostBuffer)] / kMiB,
                mem.counts[static_cast<std::uint32_t>(Kind::HostBuffer)]);
    ImGui::Text("Images: %.1f MiB (%u)",
                mem.bytes[static_cast<std::uint32_t>(Kind::Image)] / kMiB,
                mem.counts[static_cast<std::uint32_t>(Kind::Image)]);

    // GPU culling result, same treatment: what the cull dispatch emitted
    // last completed frame — draws that survived, the triangles they
    // carry (post-LOD; the graph makes LOD/occlusion react visibly while
    // flying), and what the occlusion test dropped.
    if (cull != nullptr) {
        const float megaTris = static_cast<float>(cull->triangles) / 1.0e6f;
        triangleHistory_[triangleHistoryOffset_] = megaTris;
        triangleHistoryOffset_ = (triangleHistoryOffset_ + 1) % triangleHistory_.size();
        ImGui::Separator();
        ImGui::Text("Culling: %u / %u draws, %u occluded", cull->drawsInView,
                    cull->drawsLive, cull->occluded);
        ImGui::Text("Triangles: %.2fM", megaTris);
        ImGui::PlotLines("##triHistory", triangleHistory_.data(),
                         static_cast<int>(triangleHistory_.size()),
                         static_cast<int>(triangleHistoryOffset_), nullptr, 0.0f, FLT_MAX,
                         ImVec2(180.0f, 42.0f));
    }
    ImGui::End();

    // Scene asset loading bar. Wait mode covers the whole viewport (the
    // scene stays hidden behind it until every asset landed); streaming
    // mode floats a small bar top-center over the already-visible scene.
    if (loading != nullptr && loading->total > 0) {
        const float fraction =
            static_cast<float>(loading->done) / static_cast<float>(loading->total);
        char label[64];
        std::snprintf(label, sizeof(label), "Loading textures %u / %u", loading->done,
                      loading->total);
        if (loading->hideScene) {
            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
            ImGui::SetNextWindowSize(io.DisplaySize);
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.03f, 0.01f, 0.01f, 1.0f));
            ImGui::Begin("##loadingCover", nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoBringToFrontOnFocus);
            const float barWidth = std::min(420.0f, io.DisplaySize.x * 0.6f);
            ImGui::SetCursorPos(ImVec2((io.DisplaySize.x - barWidth) * 0.5f,
                                       io.DisplaySize.y * 0.5f - 24.0f));
            ImGui::TextUnformatted(label);
            ImGui::SetCursorPos(
                ImVec2((io.DisplaySize.x - barWidth) * 0.5f, io.DisplaySize.y * 0.5f));
            ImGui::ProgressBar(fraction, ImVec2(barWidth, 18.0f));
            ImGui::End();
            ImGui::PopStyleColor();
        } else {
            const float barWidth = 280.0f;
            ImGui::SetNextWindowPos(ImVec2((io.DisplaySize.x - barWidth) * 0.5f, 8.0f));
            ImGui::Begin("##loadingBar", nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
            ImGui::TextUnformatted(label);
            ImGui::ProgressBar(fraction, ImVec2(barWidth, 12.0f));
            ImGui::End();
        }
    }

    frameBuilt_ = true;
}

void Ui::render(VkCommandBuffer cmd) {
    if (!frameBuilt_) {
        return; // drawFrame without a built frame (shouldn't happen)
    }
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    frameBuilt_ = false;
}

} // namespace viewer

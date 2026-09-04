#include "ui.h"

#include "rend/core/log.h"
#include "rend/gpu/device.h"
#include "rend/gpu/instance.h"
#include "rend/gpu/swapchain.h"

#include <volk.h>

#include <imgui.h>
#include <imgui_impl_vulkan.h>

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

void Ui::buildFrame(std::uint32_t width, std::uint32_t height, float deltaSeconds) {
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(width), static_cast<float>(height));
    io.DeltaTime = deltaSeconds > 0.0f ? deltaSeconds : 1.0f / 60.0f;

    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();

    // FPS counter: text only (no background, no borders), top-left corner,
    // smoothed so it is readable instead of flickering.
    smoothedFrameSeconds_ = smoothedFrameSeconds_ <= 0.0f
                                ? io.DeltaTime
                                : smoothedFrameSeconds_ * 0.95f + io.DeltaTime * 0.05f;
    ImGui::SetNextWindowPos(ImVec2(8.0f, 8.0f));
    ImGui::Begin("##fps", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground |
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoInputs |
                     ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoSavedSettings);
    ImGui::Text("%.0f FPS (%.2f ms)", 1.0f / smoothedFrameSeconds_,
                smoothedFrameSeconds_ * 1000.0f);
    ImGui::End();

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

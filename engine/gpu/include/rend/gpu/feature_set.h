#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace rend::gpu {

// Device capabilities the engine understands, declared as data. Each maps
// to one or more Vulkan feature bits internally (see device.cpp). This is
// the root of the offer chain: what the hardware supports decides what the
// backend enables, which decides what the renderer can offer the game.
enum class Feature : std::uint32_t {
    MultiDrawIndirect,         // core: multiDrawIndirect
    DrawIndirectFirstInstance, // core: drawIndirectFirstInstance
    ShaderDrawParameters,      // 1.1: gl_DrawID etc.
    DescriptorIndexing,        // 1.2: bindless descriptor arrays
    DrawIndirectCount,         // 1.2: vkCmdDrawIndexedIndirectCount
    TimelineSemaphore,         // 1.2: async transfer/defrag sync
    BufferDeviceAddress,       // 1.2: GPU pointers
    DynamicRendering,          // 1.3: render without render passes
    Synchronization2,          // 1.3: modern barrier API
    ShaderDemote,              // 1.3: dxc compiles HLSL discard to OpDemote
    AccelerationStructure,     // ext: BLAS/TLAS build + storage
    RayQuery,                  // ext: inline ray tracing from any stage
    Count,
};

std::string_view featureName(Feature f);

// Device extension names usable without pulling in Vulkan headers.
inline constexpr const char* kSwapchainExtension = "VK_KHR_swapchain";

// A requested capability profile: required features fail device selection
// if unsupported; optional ones are enabled when available and reported.
struct FeatureSet {
    std::vector<Feature> required;
    std::vector<Feature> optional;
    std::vector<const char*> requiredExtensions;
    std::vector<const char*> optionalExtensions;

    // The profile for the static-command-buffer / GPU-driven model
    // (docs/ARCHITECTURE.md): all ubiquitous on desktop hardware.
    static FeatureSet gpuDriven();
};

} // namespace rend::gpu

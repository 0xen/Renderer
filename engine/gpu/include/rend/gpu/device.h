#pragma once

#include "rend/core/result.h"
#include "rend/gpu/feature_set.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

typedef struct VkPhysicalDevice_T* VkPhysicalDevice;
typedef struct VkDevice_T* VkDevice;
typedef struct VkQueue_T* VkQueue;

namespace rend::gpu {

class Instance;

// Shadow techniques the device can execute, ordered cheapest-first. The
// app picks from this list; it never assumes a technique exists.
enum class ShadowTechnique {
    CascadedShadowMaps, // raster depth passes; runs everywhere
    RayTraced,          // inline ray query against a BVH; needs RT features
};

const char* shadowTechniqueName(ShadowTechnique technique);

// Reflection techniques for reflective-tagged objects, ordered cheapest
// first — same offer pattern as shadows. When primary visibility itself is
// ray traced the choice is superseded: everything reflects for real.
enum class ReflectionTechnique {
    ReflectionProbe, // load-time cubemap capture; runs everywhere
    RayTraced,       // one inline reflection ray per fragment; needs RT features
};

const char* reflectionTechniqueName(ReflectionTechnique technique);

// Anti-aliasing techniques for the post chain, ordered cheapest first —
// the same offer pattern as shadows/reflections. The app picks one and
// wires the matching module (a pipeline stage after the post pass) into
// the DrawBatch; None (the "off" choice) always exists.
enum class AntiAliasingTechnique {
    None, // post pass straight to the swapchain
    Fxaa, // fullscreen LDR fragment pass over the post output; runs everywhere
};

const char* antiAliasingTechniqueName(AntiAliasingTechnique technique);

struct QueueInfo {
    VkQueue queue = nullptr;
    std::uint32_t familyIndex = ~0u;
    bool valid() const { return queue != nullptr; }
};

// Owns adapter selection (scored against the FeatureSet), the VkDevice,
// and its queues. Reports which optional features were actually enabled —
// the source the renderer's capability offers derive from.
class Device {
public:
    static Result<std::unique_ptr<Device>> create(const Instance& instance, const FeatureSet& features);
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    VkDevice handle() const { return device_; }
    VkPhysicalDevice physicalDevice() const { return physical_; }
    const std::string& adapterName() const { return adapterName_; }

    // Graphics queue also handles compute + present.
    const QueueInfo& graphicsQueue() const { return graphics_; }
    // Dedicated transfer-only family when the hardware has one (async
    // streaming/defrag); falls back to the graphics queue otherwise.
    const QueueInfo& transferQueue() const { return transfer_; }
    bool hasDedicatedTransfer() const { return transfer_.familyIndex != graphics_.familyIndex; }

    bool isEnabled(Feature f) const { return (enabledMask_ & (1ull << static_cast<std::uint32_t>(f))) != 0; }

    // Adapter's max sampler anisotropy when the feature was enabled at
    // creation, 0 when unavailable (samplers must stay isotropic then).
    float maxSamplerAnisotropy() const { return maxSamplerAnisotropy_; }

    // Capability offer: which shadow techniques this device can run,
    // derived from the features that were actually enabled.
    std::vector<ShadowTechnique> supportedShadowTechniques() const;
    // Which reflection techniques this device can run; the probe tier is
    // the floor and exists everywhere.
    std::vector<ReflectionTechnique> supportedReflectionTechniques() const;
    // Which anti-aliasing techniques this device can run. None and FXAA
    // need nothing beyond the baseline; future temporal/compute modules
    // gate on their features here.
    std::vector<AntiAliasingTechnique> supportedAntiAliasingTechniques() const;

private:
    Device() = default;

    VkPhysicalDevice physical_ = nullptr;
    VkDevice device_ = nullptr;
    QueueInfo graphics_;
    QueueInfo transfer_;
    std::string adapterName_;
    std::uint64_t enabledMask_ = 0;
    float maxSamplerAnisotropy_ = 0.0f;
};

} // namespace rend::gpu

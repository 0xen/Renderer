#pragma once

#include "rend/core/result.h"
#include "rend/gpu/api.h"
#include "rend/gpu/feature_set.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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

// Owns adapter selection (scored against the FeatureSet), the API device
// and its queues. Reports which optional features were actually enabled —
// the source the renderer's capability offers derive from.
class Device {
public:
    static Result<std::unique_ptr<Device>> create(const Instance& instance, const FeatureSet& features);
    virtual ~Device() = default;

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    Api api() const { return api_; }
    const std::string& adapterName() const { return adapterName_; }

    // Whether the hardware has a transfer-only queue family used for
    // uploads (async streaming/defrag later).
    virtual bool hasDedicatedTransfer() const = 0;

    bool isEnabled(Feature f) const { return (enabledMask_ & (1ull << static_cast<std::uint32_t>(f))) != 0; }

    // Adapter's max sampler anisotropy when the feature was enabled at
    // creation, 0 when unavailable (samplers must stay isotropic then).
    float maxSamplerAnisotropy() const { return maxSamplerAnisotropy_; }

    // Blocks until every queue has drained. Used around non-update-after-
    // bind descriptor rewrites and before destruction.
    virtual void waitIdle() const = 0;

    // Backend objects for integrations that must talk to the API directly
    // (the ImGui backend): VkDevice / VkPhysicalDevice / VkQueue under
    // Vulkan. Never used by the engine's own passes.
    virtual void* nativeHandle() const = 0;
    virtual void* nativePhysicalDevice() const = 0;
    virtual void* nativeGraphicsQueue() const = 0;
    virtual std::uint32_t graphicsQueueFamily() const = 0;

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

protected:
    explicit Device(Api api) : api_(api) {}

    std::string adapterName_;
    std::uint64_t enabledMask_ = 0;
    float maxSamplerAnisotropy_ = 0.0f;

private:
    Api api_;
};

} // namespace rend::gpu

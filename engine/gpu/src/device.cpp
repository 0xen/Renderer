#include "rend/gpu/device.h"

#include "rend/core/log.h"
#include "rend/gpu/instance.h"

#include <volk.h>

#include <array>
#include <cstring>
#include <optional>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi.h>
#endif

namespace rend::gpu {

namespace {

// Full 1.1/1.2/1.3 feature chain used both to probe support and to
// declare what gets enabled at device creation.
struct FeatureChain {
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    // Ray tracing extension features: chained for probing (drivers leave
    // them false when unsupported); unlinked before device creation when
    // the features stay disabled, so no unknown-extension structs reach
    // vkCreateDevice.
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuery{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};

    FeatureChain() {
        f2.pNext = &v11;
        v11.pNext = &v12;
        v12.pNext = &v13;
        v13.pNext = &accel;
        accel.pNext = &rayQuery;
    }

    void unlinkRayTracing() { v13.pNext = nullptr; }
};

bool supports(const FeatureChain& c, Feature f) {
    switch (f) {
    case Feature::MultiDrawIndirect: return c.f2.features.multiDrawIndirect;
    case Feature::DrawIndirectFirstInstance: return c.f2.features.drawIndirectFirstInstance;
    case Feature::ShaderDrawParameters: return c.v11.shaderDrawParameters;
    case Feature::DescriptorIndexing:
        return c.v12.descriptorIndexing && c.v12.runtimeDescriptorArray &&
               c.v12.shaderSampledImageArrayNonUniformIndexing &&
               c.v12.descriptorBindingPartiallyBound &&
               c.v12.descriptorBindingSampledImageUpdateAfterBind;
    case Feature::DrawIndirectCount: return c.v12.drawIndirectCount;
    case Feature::TimelineSemaphore: return c.v12.timelineSemaphore;
    case Feature::ShaderDemote: return c.v13.shaderDemoteToHelperInvocation;
    case Feature::BufferDeviceAddress: return c.v12.bufferDeviceAddress;
    case Feature::DynamicRendering: return c.v13.dynamicRendering;
    case Feature::Synchronization2: return c.v13.synchronization2;
    case Feature::AccelerationStructure: return c.accel.accelerationStructure;
    case Feature::RayQuery: return c.rayQuery.rayQuery;
    case Feature::Count: break;
    }
    return false;
}

void enable(FeatureChain& c, Feature f) {
    switch (f) {
    case Feature::MultiDrawIndirect: c.f2.features.multiDrawIndirect = VK_TRUE; break;
    case Feature::DrawIndirectFirstInstance: c.f2.features.drawIndirectFirstInstance = VK_TRUE; break;
    case Feature::ShaderDrawParameters: c.v11.shaderDrawParameters = VK_TRUE; break;
    case Feature::DescriptorIndexing:
        c.v12.descriptorIndexing = VK_TRUE;
        c.v12.runtimeDescriptorArray = VK_TRUE;
        c.v12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        c.v12.descriptorBindingPartiallyBound = VK_TRUE;
        c.v12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
        break;
    case Feature::DrawIndirectCount: c.v12.drawIndirectCount = VK_TRUE; break;
    case Feature::ShaderDemote: c.v13.shaderDemoteToHelperInvocation = VK_TRUE; break;
    case Feature::TimelineSemaphore: c.v12.timelineSemaphore = VK_TRUE; break;
    case Feature::BufferDeviceAddress: c.v12.bufferDeviceAddress = VK_TRUE; break;
    case Feature::DynamicRendering: c.v13.dynamicRendering = VK_TRUE; break;
    case Feature::Synchronization2: c.v13.synchronization2 = VK_TRUE; break;
    case Feature::AccelerationStructure: c.accel.accelerationStructure = VK_TRUE; break;
    case Feature::RayQuery: c.rayQuery.rayQuery = VK_TRUE; break;
    case Feature::Count: break;
    }
}

struct QueueFamilies {
    std::optional<std::uint32_t> graphics;
    std::optional<std::uint32_t> dedicatedTransfer;
};

QueueFamilies findQueueFamilies(VkPhysicalDevice pd) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, props.data());

    QueueFamilies out;
    for (std::uint32_t i = 0; i < count; ++i) {
        const VkQueueFlags flags = props[i].queueFlags;
        if (!out.graphics && (flags & VK_QUEUE_GRAPHICS_BIT)) {
            out.graphics = i;
        }
        // Transfer-only family (no graphics/compute) = async copy engine.
        if (!out.dedicatedTransfer && (flags & VK_QUEUE_TRANSFER_BIT) &&
            !(flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
            out.dedicatedTransfer = i;
        }
    }
    return out;
}

// LUIDs of every adapter the OS has a display attached to. Presenting from
// an adapter with no display goes through the driver's cross-adapter copy
// path, which on some drivers (seen on AMD iGPU+dGPU systems) can deadlock
// the whole device — so adapter selection strongly prefers a display owner.
std::vector<std::array<std::uint8_t, VK_LUID_SIZE>> displayAdapterLuids() {
    std::vector<std::array<std::uint8_t, VK_LUID_SIZE>> luids;
#ifdef _WIN32
    IDXGIFactory* factory = nullptr;
    if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&factory)))) {
        return luids;
    }
    IDXGIAdapter* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        IDXGIOutput* output = nullptr;
        if (adapter->EnumOutputs(0, &output) != DXGI_ERROR_NOT_FOUND && output) {
            DXGI_ADAPTER_DESC desc{};
            if (SUCCEEDED(adapter->GetDesc(&desc))) {
                std::array<std::uint8_t, VK_LUID_SIZE> luid{};
                static_assert(sizeof(desc.AdapterLuid) == VK_LUID_SIZE);
                std::memcpy(luid.data(), &desc.AdapterLuid, VK_LUID_SIZE);
                luids.push_back(luid);
            }
            output->Release();
        }
        adapter->Release();
    }
    factory->Release();
#endif
    return luids;
}

bool hasExtension(const std::vector<VkExtensionProperties>& available, const char* name) {
    for (const auto& e : available) {
        if (std::strcmp(e.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

struct Candidate {
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    QueueFamilies families;
    FeatureChain supported;
    std::vector<const char*> optionalExts;
    std::uint64_t optionalMask = 0;
    bool drivesDisplay = false;
    int score = -1; // <0 = unsuitable
};

Candidate evaluate(VkPhysicalDevice pd, const FeatureSet& request,
                   const std::vector<std::array<std::uint8_t, VK_LUID_SIZE>>& displayLuids) {
    Candidate c;
    c.pd = pd;
    VkPhysicalDeviceIDProperties idProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &idProps;
    vkGetPhysicalDeviceProperties2(pd, &props2);
    c.props = props2.properties;
    if (c.props.apiVersion < VK_API_VERSION_1_3) {
        return c;
    }
    if (idProps.deviceLUIDValid) {
        for (const auto& luid : displayLuids) {
            if (std::memcmp(luid.data(), idProps.deviceLUID, VK_LUID_SIZE) == 0) {
                c.drivesDisplay = true;
                break;
            }
        }
    }

    c.families = findQueueFamilies(pd);
    if (!c.families.graphics) {
        return c;
    }

    vkGetPhysicalDeviceFeatures2(pd, &c.supported.f2);
    for (Feature f : request.required) {
        if (!supports(c.supported, f)) {
            log::trace("{}: missing required feature {}", c.props.deviceName, featureName(f));
            return c;
        }
    }

    std::uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, exts.data());
    for (const char* name : request.requiredExtensions) {
        if (!hasExtension(exts, name)) {
            log::trace("{}: missing required extension {}", c.props.deviceName, name);
            return c;
        }
    }

    c.score = 0;
    // Owning a display outweighs everything else: presenting from a
    // display-less adapter uses the cross-adapter path (see above).
    if (c.drivesDisplay) {
        c.score += 10000;
    }
    if (c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        c.score += 1000;
    }
    if (c.families.dedicatedTransfer) {
        c.score += 50;
    }
    for (Feature f : request.optional) {
        if (supports(c.supported, f)) {
            c.optionalMask |= 1ull << static_cast<std::uint32_t>(f);
            c.score += 10;
        }
    }
    for (const char* name : request.optionalExtensions) {
        if (hasExtension(exts, name)) {
            c.optionalExts.push_back(name);
            c.score += 10;
        }
    }
    return c;
}

} // namespace

std::string_view featureName(Feature f) {
    switch (f) {
    case Feature::MultiDrawIndirect: return "MultiDrawIndirect";
    case Feature::DrawIndirectFirstInstance: return "DrawIndirectFirstInstance";
    case Feature::ShaderDrawParameters: return "ShaderDrawParameters";
    case Feature::DescriptorIndexing: return "DescriptorIndexing";
    case Feature::DrawIndirectCount: return "DrawIndirectCount";
    case Feature::ShaderDemote: return "ShaderDemote";
    case Feature::TimelineSemaphore: return "TimelineSemaphore";
    case Feature::BufferDeviceAddress: return "BufferDeviceAddress";
    case Feature::DynamicRendering: return "DynamicRendering";
    case Feature::Synchronization2: return "Synchronization2";
    case Feature::AccelerationStructure: return "AccelerationStructure";
    case Feature::RayQuery: return "RayQuery";
    case Feature::Count: break;
    }
    return "Unknown";
}

FeatureSet FeatureSet::gpuDriven() {
    return {
        .required = {Feature::ShaderDrawParameters, Feature::DescriptorIndexing,
                     Feature::TimelineSemaphore, Feature::DynamicRendering,
                     Feature::Synchronization2, Feature::ShaderDemote},
        // The indirect-draw ladder is optional: the scene pass detects what
        // is enabled and falls back (indirect-count -> multi-draw indirect
        // -> per-draw vkCmdDrawIndexed), so a device missing these still
        // renders, just without the static-buffer wins. Ray tracing is an
        // offer on top of the shadow-map floor (docs/ARCHITECTURE.md).
        .optional = {Feature::MultiDrawIndirect, Feature::DrawIndirectFirstInstance,
                     Feature::DrawIndirectCount, Feature::BufferDeviceAddress,
                     Feature::AccelerationStructure, Feature::RayQuery},
        .requiredExtensions = {},
        .optionalExtensions = {},
    };
}

Result<std::unique_ptr<Device>> Device::create(const Instance& instance, const FeatureSet& request) {
    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance.handle(), &count, nullptr);
    if (count == 0) {
        return Error{"No Vulkan-capable GPU found"};
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance.handle(), &count, devices.data());

    const auto displayLuids = displayAdapterLuids();
    if (displayLuids.empty()) {
        log::warn("Could not determine which adapter drives the display; scoring by type only");
    }

    Candidate best;
    for (VkPhysicalDevice pd : devices) {
        Candidate c = evaluate(pd, request, displayLuids);
        log::info("Adapter: {} (api {}.{}, {}{}) — {}", c.props.deviceName,
                  VK_API_VERSION_MAJOR(c.props.apiVersion), VK_API_VERSION_MINOR(c.props.apiVersion),
                  c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "discrete" : "integrated/other",
                  c.drivesDisplay ? ", drives a display" : ", no display",
                  c.score < 0 ? "unsuitable" : std::format("score {}", c.score));
        if (c.score > best.score) {
            best = c;
        }
    }
    if (best.score < 0) {
        return Error{"No adapter satisfies the required feature set"};
    }

    // Queue create infos: graphics always; dedicated transfer when distinct.
    const float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    VkDeviceQueueCreateInfo qi{};
    qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi.queueFamilyIndex = *best.families.graphics;
    qi.queueCount = 1;
    qi.pQueuePriorities = &priority;
    queueInfos.push_back(qi);
    if (best.families.dedicatedTransfer) {
        qi.queueFamilyIndex = *best.families.dedicatedTransfer;
        queueInfos.push_back(qi);
    }

    FeatureChain enabled;
    std::uint64_t mask = 0;
    for (Feature f : request.required) {
        enable(enabled, f);
        mask |= 1ull << static_cast<std::uint32_t>(f);
    }
    for (Feature f : request.optional) {
        if (best.optionalMask & (1ull << static_cast<std::uint32_t>(f))) {
            enable(enabled, f);
            mask |= 1ull << static_cast<std::uint32_t>(f);
        } else {
            log::warn("Optional feature {} unavailable on {}", featureName(f), best.props.deviceName);
        }
    }

    std::vector<const char*> extensions = request.requiredExtensions;
    extensions.insert(extensions.end(), best.optionalExts.begin(), best.optionalExts.end());

    // Ray tracing rides on device extensions; enable them alongside the
    // features (their support is implied by the feature probe), or unlink
    // the extension structs so vkCreateDevice never sees them.
    const bool rt = (mask & (1ull << static_cast<std::uint32_t>(Feature::AccelerationStructure))) &&
                    (mask & (1ull << static_cast<std::uint32_t>(Feature::RayQuery)));
    if (rt) {
        extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    } else {
        // Both or neither: a lone half of the pair is useless and its
        // struct never reaches vkCreateDevice.
        enabled.unlinkRayTracing();
        mask &= ~(1ull << static_cast<std::uint32_t>(Feature::AccelerationStructure));
        mask &= ~(1ull << static_cast<std::uint32_t>(Feature::RayQuery));
    }

    VkDeviceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.pNext = &enabled.f2;
    info.queueCreateInfoCount = static_cast<std::uint32_t>(queueInfos.size());
    info.pQueueCreateInfos = queueInfos.data();
    info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();

    VkDevice handle = VK_NULL_HANDLE;
    if (VkResult r = vkCreateDevice(best.pd, &info, nullptr, &handle); r != VK_SUCCESS) {
        return Error{std::format("vkCreateDevice failed ({})", static_cast<int>(r))};
    }
    volkLoadDevice(handle);

    auto device = std::unique_ptr<Device>(new Device());
    device->physical_ = best.pd;
    device->device_ = handle;
    device->adapterName_ = best.props.deviceName;
    device->enabledMask_ = mask;

    device->graphics_.familyIndex = *best.families.graphics;
    vkGetDeviceQueue(handle, device->graphics_.familyIndex, 0, &device->graphics_.queue);
    if (best.families.dedicatedTransfer) {
        device->transfer_.familyIndex = *best.families.dedicatedTransfer;
        vkGetDeviceQueue(handle, device->transfer_.familyIndex, 0, &device->transfer_.queue);
    } else {
        device->transfer_ = device->graphics_;
    }

    log::info("Device created on '{}' (graphics family {}, transfer family {}{})", device->adapterName_,
              device->graphics_.familyIndex, device->transfer_.familyIndex,
              device->hasDedicatedTransfer() ? ", dedicated" : ", shared");
    return device;
}

Device::~Device() {
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        log::info("Device destroyed ('{}')", adapterName_);
    }
}

} // namespace rend::gpu

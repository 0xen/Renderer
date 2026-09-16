#include "rend/gpu/instance.h"

#include "vulkan/vulkan_types.h"
#include "d3d12/d3d12_types.h"

#include "rend/core/log.h"

#include <volk.h>

#include <cstring>
#include <format>

namespace rend::gpu {

namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data,
                                             void* /*userData*/) {
    const char* text = data && data->pMessage ? data->pMessage : "(no message)";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        log::error("[vulkan] {}", text);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        log::warn("[vulkan] {}", text);
    } else {
        log::trace("[vulkan] {}", text);
    }
    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT makeMessengerInfo() {
    VkDebugUtilsMessengerCreateInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debugCallback;
    return info;
}

bool hasLayer(const char* name) {
    std::uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& layer : layers) {
        if (std::strcmp(layer.layerName, name) == 0) {
            return true;
        }
    }
    return false;
}

} // namespace

Result<std::unique_ptr<Instance>> VulkanInstance::create(const InstanceDesc& desc) {
    static bool volkReady = false;
    if (!volkReady) {
        if (volkInitialize() != VK_SUCCESS) {
            return Error{"volkInitialize failed — is a Vulkan loader/driver installed?"};
        }
        volkReady = true;
    }

    std::uint32_t supported = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion) {
        vkEnumerateInstanceVersion(&supported);
    }
    if (supported < VK_API_VERSION_1_3) {
        return Error{std::format("Vulkan 1.3 required, loader supports only {}.{}",
                                 VK_API_VERSION_MAJOR(supported), VK_API_VERSION_MINOR(supported))};
    }

    const bool validation = desc.enableValidation && hasLayer(kValidationLayer);
    if (desc.enableValidation && !validation) {
        log::warn("Validation requested but {} not installed; continuing without it", kValidationLayer);
    }

    std::vector<const char*> extensions = desc.extraExtensions;
    std::vector<const char*> layers;
    if (validation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        layers.push_back(kValidationLayer);
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = desc.appName.c_str();
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.pEngineName = "rend";
    app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    // Chain a messenger info into instance creation so create/destroy of
    // the instance itself is also covered by validation.
    VkDebugUtilsMessengerCreateInfoEXT messengerInfo = makeMessengerInfo();

    // Synchronization validation is opt-in via VK_EXT_validation_features,
    // chained through the messenger so both reach the layer.
    const bool syncValidation = validation && desc.enableSyncValidation;
    VkValidationFeatureEnableEXT syncEnable = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    VkValidationFeaturesEXT validationFeatures{};
    validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
    validationFeatures.enabledValidationFeatureCount = 1;
    validationFeatures.pEnabledValidationFeatures = &syncEnable;
    if (syncValidation) {
        messengerInfo.pNext = &validationFeatures;
    }

    VkInstanceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pNext = validation ? &messengerInfo : nullptr;
    info.pApplicationInfo = &app;
    info.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    info.ppEnabledLayerNames = layers.data();
    info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();

    VkInstance handle = VK_NULL_HANDLE;
    if (VkResult r = vkCreateInstance(&info, nullptr, &handle); r != VK_SUCCESS) {
        return Error{std::format("vkCreateInstance failed ({})", static_cast<int>(r))};
    }
    volkLoadInstance(handle);

    auto instance = std::unique_ptr<VulkanInstance>(new VulkanInstance());
    instance->instance_ = handle;
    instance->apiVersion_ = supported;

    if (validation) {
        if (VkResult r = vkCreateDebugUtilsMessengerEXT(handle, &messengerInfo, nullptr,
                                                        &instance->messenger_);
            r != VK_SUCCESS) {
            log::warn("vkCreateDebugUtilsMessengerEXT failed ({}); continuing without messenger",
                      static_cast<int>(r));
            instance->messenger_ = VK_NULL_HANDLE;
        }
    }

    log::info("Vulkan instance created (loader {}.{}.{}, validation {}, sync validation {})",
              VK_API_VERSION_MAJOR(supported), VK_API_VERSION_MINOR(supported),
              VK_API_VERSION_PATCH(supported), validation ? "on" : "off",
              syncValidation ? "on" : "off");
    return std::unique_ptr<Instance>(std::move(instance));
}

VulkanInstance::~VulkanInstance() {
    if (messenger_ != VK_NULL_HANDLE && vkDestroyDebugUtilsMessengerEXT) {
        vkDestroyDebugUtilsMessengerEXT(instance_, messenger_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        log::info("Vulkan instance destroyed");
    }
}

Result<std::unique_ptr<Instance>> Instance::create(Api api, const InstanceDesc& desc) {
    switch (api) {
    case Api::Vulkan: return VulkanInstance::create(desc);
    case Api::D3D12: return D3D12Instance::create(desc);
    }
    return Error{std::format("{} backend: not implemented", apiName(api))};
}

} // namespace rend::gpu

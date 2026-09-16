#pragma once

namespace rend::gpu {

// Which graphics API a gpu-layer object tree was created for. Chosen once
// at Instance::create; every object created from that instance's device
// belongs to the same backend and the static factories dispatch on it.
enum class Api {
    Vulkan,
    D3D12,
};

inline constexpr const char* apiName(Api api) {
    switch (api) {
    case Api::Vulkan: return "Vulkan";
    case Api::D3D12: return "D3D12";
    }
    return "?";
}

} // namespace rend::gpu

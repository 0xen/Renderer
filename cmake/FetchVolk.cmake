# Fetches volk (Vulkan function loader), pinned to the tag matching the
# installed Vulkan SDK. volk pulls Vulkan headers from the SDK via VULKAN_SDK.
include(FetchContent)

FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG vulkan-sdk-1.4.357.0
    GIT_SHALLOW ON
)
FetchContent_MakeAvailable(volk)

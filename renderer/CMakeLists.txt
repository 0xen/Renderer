set(project_name Renderer)
project(${project_name})

find_package(Vulkan)

set(common_src
	src/renderer/IRenderer.cpp
)

set(common_headers
    include/renderer/NativeWindowHandle.hpp
    include/renderer/IRenderer.hpp
    include/renderer/Status.hpp
    include/renderer/APIs.hpp
)

set(source
    ${common_src}
)

set(headers
    ${common_headers}
)

find_package(Vulkan)

if(Vulkan_FOUND)
    
    set(vulkan_src
       src/renderer/vulkan/VulkanCommon.cpp
       src/renderer/vulkan/VulkanInitializers.cpp
       src/renderer/vulkan/VulkanInstance.cpp
       src/renderer/vulkan/VulkanRenderer.cpp
       src/renderer/vulkan/VulkanPhysicalDevice.cpp
       src/renderer/vulkan/VulkanDevice.cpp
       src/renderer/vulkan/VulkanSwapchain.cpp
    )

    set(vulkan_headers
        include/renderer/vulkan/VulkanSwapChainSupport.hpp
        include/renderer/vulkan/VulkanHeader.hpp
        include/renderer/vulkan/VulkanStatus.hpp
        include/renderer/vulkan/VulkanCommon.hpp
        include/renderer/vulkan/VulkanInitializers.hpp
        include/renderer/vulkan/VulkanInstance.hpp
        include/renderer/vulkan/VulkanRenderer.hpp
        include/renderer/vulkan/VulkanQueueFamilyIndices.hpp
        include/renderer/vulkan/VulkanPhysicalDevice.hpp
        include/renderer/vulkan/VulkanDevice.hpp
        include/renderer/vulkan/VulkanSwapchain.hpp
    )

    include_directories(${Vulkan_INCLUDE_DIRS})

    list(APPEND source ${vulkan_src})
    list(APPEND headers ${vulkan_headers})

    source_group("Source Files\\RenderingAPI\\Vulkan" FILES ${vulkan_src})
    source_group("Header Files\\RenderingAPI\\Vulkan" FILES ${vulkan_headers})


endif()
include_directories(include)

add_library(${project_name} STATIC ${source} ${headers})

target_link_libraries(${project_name} SDL2-static)



if(Vulkan_FOUND)
    
    target_link_libraries(${project_name} Vulkan::Vulkan)

endif()
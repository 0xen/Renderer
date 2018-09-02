set(project_name Renderer)
project(${project_name})

set(common_src
    src/renderer/IRenderer.cpp
    src/renderer/IBuffer.cpp
    src/renderer/IUniformBuffer.cpp
    src/renderer/IPipeline.cpp
    src/renderer/IComputePipeline.cpp
    src/renderer/IComputeProgram.cpp
    src/renderer/IVertexBuffer.cpp
    src/renderer/VertexBase.cpp
    src/renderer/IIndexBuffer.cpp
    src/renderer/IGraphicsPipeline.cpp
    src/renderer/IModel.cpp
    src/renderer/IModelPool.cpp
    src/renderer/IDescriptor.cpp
    src/renderer/IDescriptorPool.cpp
    src/renderer/IDescriptorSet.cpp
)

set(common_headers
    include/renderer/NativeWindowHandle.hpp
    include/renderer/IRenderer.hpp
    include/renderer/IBuffer.hpp
    include/renderer/IUniformBuffer.hpp
    include/renderer/Status.hpp
    include/renderer/APIs.hpp
    include/renderer/ShaderStage.hpp
    include/renderer/DescriptorType.hpp
    include/renderer/IPipeline.hpp
    include/renderer/IComputePipeline.hpp
    include/renderer/IComputeProgram.hpp
    include/renderer/IVertexBuffer.hpp
    include/renderer/IIndexBuffer.hpp
    include/renderer/IGraphicsPipeline.hpp
    include/renderer/VertexBase.hpp
    include/renderer/VertexBinding.hpp
    include/renderer/DataFormat.hpp
    include/renderer/IModel.hpp
    include/renderer/IModelPool.hpp
    include/renderer/VertexInputRate.hpp
    include/renderer/ITextureBuffer.hpp
    include/renderer/IDescriptor.hpp
    include/renderer/IDescriptorPool.hpp
    include/renderer/IDescriptorSet.hpp

)

set(source
    ${common_src}
)

set(headers
    ${common_headers}
)

# Load imgui
set(imgui_src
    ../third_party/imgui/imgui.cpp
    ../third_party/imgui/imgui_demo.cpp
    ../third_party/imgui/imgui_draw.cpp
    ../third_party/imgui/imgui_widgets.cpp
)
set(imgui_headers
    ../third_party/imgui/imconfig.h
    ../third_party/imgui/imgui.h
    ../third_party/imgui/imgui_internal.h
    ../third_party/imgui/stb_rect_pack.h
    ../third_party/imgui/stb_textedit.h
    ../third_party/imgui/stb_truetype.h
)
list(APPEND source ${imgui_src})
list(APPEND headers ${imgui_headers})

source_group("Source Files\\External\\imgui" FILES ${imgui_src})
source_group("Header Files\\External\\imgui" FILES ${imgui_headers})


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
        src/renderer/vulkan/VulkanBuffer.cpp
        src/renderer/vulkan/VulkanUniformBuffer.cpp
        src/renderer/vulkan/VulkanPipeline.cpp
        src/renderer/vulkan/VulkanComputePipeline.cpp
        src/renderer/vulkan/VulkanComputeProgram.cpp
        src/renderer/vulkan/VulkanVertexBuffer.cpp
        src/renderer/vulkan/VulkanIndexBuffer.cpp
        src/renderer/vulkan/VulkanGraphicsPipeline.cpp
        src/renderer/vulkan/VulkanModel.cpp
        src/renderer/vulkan/VulkanModelPool.cpp
        src/renderer/vulkan/VulkanTextureBuffer.cpp
        src/renderer/vulkan/VulkanDescriptor.cpp
        src/renderer/vulkan/VulkanDescriptorPool.cpp
        src/renderer/vulkan/VulkanDescriptorSet.cpp
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
        include/renderer/vulkan/VulkanBuffer.hpp
        include/renderer/vulkan/VulkanUniformBuffer.hpp
        include/renderer/vulkan/VulkanBufferData.hpp
        include/renderer/vulkan/VulkanPipeline.hpp
        include/renderer/vulkan/VulkanComputePipeline.hpp
        include/renderer/vulkan/VulkanComputeProgram.hpp
        include/renderer/vulkan/VulkanVertexBuffer.hpp
        include/renderer/vulkan/VulkanIndexBuffer.hpp
        include/renderer/vulkan/VulkanGraphicsPipeline.hpp
        include/renderer/vulkan/VulkanModel.hpp
        include/renderer/vulkan/VulkanModelPool.hpp
        include/renderer/vulkan/VulkanTextureBuffer.hpp
        include/renderer/vulkan/VulkanDescriptor.hpp
        include/renderer/vulkan/VulkanDescriptorPool.hpp
        include/renderer/vulkan/VulkanDescriptorSet.hpp
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
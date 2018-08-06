set(project_name Renderer)
project(${project_name})

set(common_source
	src/IRenderer.cpp
)

set(common_headers
	include/IRenderer.hpp
)

include_directories(include)

set(source
    ${common_source}
)

set(headers
    ${common_headers}
)

find_package(Vulkan)

if(Vulkan_FOUND)
    add_definitions(-DVIKING_SUPPORTS_VULKAN)

    if(WIN32)
        add_definitions(-DVK_USE_PLATFORM_WIN32_KHR)
    else()
        message(FATAL "Vulkan is not currently supported on non windows platforms... Sorry :(")
    endif()

    set(vulkan_source

    )

    set(vulkan_headers

    )

    include_directories(${Vulkan_INCLUDE_DIRS})

    list(APPEND source ${vulkan_source})
    list(APPEND headers ${vulkan_headers})

    #source_group("Source Files\\Vulkan Backend" FILES ${vulkan_source})
    #source_group("Header Files\\Vulkan Backend" FILES ${vulkan_headers})
endif()

add_library(${project_name} STATIC ${source} ${headers})

target_link_libraries(${project_name} SDL2-static)
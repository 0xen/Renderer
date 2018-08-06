set(project_name renderer-demo)
project(${project_name})

set(src
    src/main.cpp
)

set(headers
)

include_directories(include)
include_directories(../third_party/glm)

add_executable(${project_name} ${src} ${headers})

target_link_libraries(${project_name} SDL2main)


find_package(Vulkan)

if(Vulkan_FOUND)
	target_include_directories(${project_name} PRIVATE Vulkan::Vulkan)
	target_link_libraries(${project_name} Vulkan::Vulkan)
endif()
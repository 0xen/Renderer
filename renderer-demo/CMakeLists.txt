set(project_name renderer-demo)
project(${project_name})

set(src
    src/main.cpp
)

set(headers
)

set(HAVE_LIBC TRUE)

# Load Lodepng
set(lodepng_src
    ../../lodepng/lodepng.cpp
)
set(lodepng_headers
    ../../lodepng/lodepng.h
)
list(APPEND src ${lodepng_src})
list(APPEND headers ${lodepng_headers})




include_directories(include)
include_directories(../third_party/glm)
include_directories(../renderer/include)

add_executable(${project_name} ${src} ${headers})

target_link_libraries(${project_name} Renderer)
target_link_libraries(${project_name} SDL2main)
target_link_libraries(${project_name} SDL2-static)


find_package(Vulkan)

if(Vulkan_FOUND)
	target_include_directories(${project_name} PRIVATE Vulkan::Vulkan)
	target_link_libraries(${project_name} Vulkan::Vulkan)
endif()



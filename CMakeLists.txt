cmake_minimum_required(VERSION 2.6)

set(project_name demo_VK-EZ)
project(${project_name})

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(HAVE_LIBC TRUE)

add_subdirectory(third_party/SDL2)

set_target_properties (SDL2 PROPERTIES FOLDER "Third Party/SDL2")
set_target_properties (SDL2main PROPERTIES FOLDER "Third Party/SDL2")
set_target_properties (SDL2-static PROPERTIES FOLDER "Third Party/SDL2")
set_target_properties (uninstall PROPERTIES FOLDER "Third Party/SDL2")

include_directories(third_party/SDL2/include)
include_directories(../lodepng)


add_subdirectory(third_party/glm)
include_directories(third_party/glm)


add_subdirectory(renderer)
add_subdirectory(renderer-demo)
add_subdirectory(renderer-demo-rasterizer)


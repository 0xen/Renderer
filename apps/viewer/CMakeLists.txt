add_executable(viewer main.cpp ui.cpp)
target_link_libraries(viewer PRIVATE rend::engine rend::assetio rend_warnings imgui)
add_dependencies(viewer shaders)
set_target_properties(viewer PROPERTIES FOLDER "apps")

# Per-scene launcher .bat files (double-clickable, CRLF — LF-only batch
# files misparse) copied next to the exe so they survive clean builds.
file(GLOB REND_VIEWER_LAUNCHERS "${CMAKE_CURRENT_SOURCE_DIR}/launchers/*.bat")
add_custom_command(TARGET viewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${REND_VIEWER_LAUNCHERS} "$<TARGET_FILE_DIR:viewer>"
    COMMENT "Copying scene launcher scripts")

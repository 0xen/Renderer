# Python host — a standalone optional DLL (assetio pattern) that embeds
# CPython and exposes the renderer message queue to scripts as a `rend`
# module. Nothing links against it: the viewer LoadLibrary's it only when
# a scene declares <Script> nodes, so scenes without scripts (or installs
# without the DLL) involve no Python at all.
add_library(rend_pyhost SHARED
    src/pyhost.cpp
)
add_library(rend::pyhost ALIAS rend_pyhost)

target_include_directories(rend_pyhost PUBLIC include)
target_link_libraries(rend_pyhost
    PUBLIC rend::engine
    PRIVATE rend_warnings pybind11::embed
)
# Dev builds bake in the fetched CPython's home (python.exe/Lib live
# there); PYTHONHOME overrides at runtime for deployed trees.
target_compile_definitions(rend_pyhost PRIVATE
    REND_PYHOST_PYTHON_HOME="${REND_PYTHON_HOME}"
    _CRT_SECURE_NO_WARNINGS # _wgetenv for the PYTHONHOME override
)
set_target_properties(rend_pyhost PROPERTIES
    OUTPUT_NAME "rend_pyhost"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    FOLDER "pyhost"
)

# python3xx.dll must resolve when the viewer LoadLibrary's this DLL.
add_custom_command(TARGET rend_pyhost POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${REND_PYTHON_RUNTIME_DLLS} $<TARGET_FILE_DIR:rend_pyhost>
)

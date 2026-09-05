# Fetches the official CPython Windows build from NuGet, pinned. The
# package is a plain zip carrying everything embedding needs — headers,
# import lib, python313.dll, and the stdlib — so building the Python host
# requires NO system Python install. Only included when REND_PYTHON is ON.
set(REND_PYTHON_VERSION 3.13.15)
set(REND_PYTHON_NUPKG_URL
    "https://api.nuget.org/v3-flatcontainer/python/${REND_PYTHON_VERSION}/python.${REND_PYTHON_VERSION}.nupkg")

set(_python_root ${CMAKE_BINARY_DIR}/_deps/cpython-${REND_PYTHON_VERSION})
if(NOT EXISTS ${_python_root}/tools/python.exe)
    message(STATUS "Fetching CPython ${REND_PYTHON_VERSION} (NuGet)...")
    set(_python_nupkg ${CMAKE_BINARY_DIR}/_deps/python-${REND_PYTHON_VERSION}.nupkg)
    file(DOWNLOAD ${REND_PYTHON_NUPKG_URL} ${_python_nupkg}
        STATUS _python_download_status
        SHOW_PROGRESS)
    list(GET _python_download_status 0 _python_download_code)
    if(NOT _python_download_code EQUAL 0)
        message(FATAL_ERROR "CPython download failed: ${_python_download_status}")
    endif()
    file(ARCHIVE_EXTRACT INPUT ${_python_nupkg} DESTINATION ${_python_root})
    file(REMOVE ${_python_nupkg})
endif()

# The runtime home for the embedded interpreter: tools/ holds python.exe,
# python3xx.dll and Lib/ (the stdlib). The host DLL bakes this path in for
# dev builds; PYTHONHOME overrides it (deployment copies the tree).
set(REND_PYTHON_HOME ${_python_root}/tools)

# Hints for FindPython (used directly and via pybind11).
set(Python_ROOT_DIR ${REND_PYTHON_HOME})
set(Python3_ROOT_DIR ${REND_PYTHON_HOME})
set(Python_EXECUTABLE ${REND_PYTHON_HOME}/python.exe)

find_package(Python 3.13 EXACT REQUIRED COMPONENTS Interpreter Development.Embed)

# python3xx.dll must sit next to viewer.exe for rend_pyhost.dll to resolve.
file(GLOB REND_PYTHON_RUNTIME_DLLS ${REND_PYTHON_HOME}/python*.dll)

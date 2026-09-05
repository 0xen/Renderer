# Fetches pybind11, pinned. Header-only; used by the Python host DLL to
# embed CPython and expose the renderer message queue as a `rend` module.
# PYBIND11_FINDPYTHON makes it use modern FindPython, which honors the
# Python_ROOT_DIR hint from FetchPython.cmake (the NuGet CPython).
include(FetchContent)

set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
    GIT_SHALLOW ON
)
FetchContent_MakeAvailable(pybind11)

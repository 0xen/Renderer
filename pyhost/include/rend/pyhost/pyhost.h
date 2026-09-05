#pragma once

// C ABI of the Python host DLL. The viewer never links rend_pyhost —
// it LoadLibrary's the DLL and resolves these three entry points, so a
// build (or install) without Python bits still runs every scene that
// declares no scripts. Keep this header free of Python and pybind11.

namespace rend::renderer {
class MessageQueue;
}

extern "C" {

// Starts the embedded interpreter on its own thread and runs the given
// script files in order against the queue (each script imports `rend`).
// The queue must outlive the host; call rendPyHostStop() before tearing
// it down. Returns false if a host is already running or args are bad.
bool rendPyHostStart(rend::renderer::MessageQueue* queue, const char* const* scriptPaths,
                     int scriptCount);

// Asks the interpreter to stop (rend.should_quit() flips, and a
// KeyboardInterrupt is raised to break sleeps), then joins the thread.
// Safe to call when nothing is running.
void rendPyHostStop(void);

// True while the script thread is alive (scripts still executing).
bool rendPyHostRunning(void);
}

namespace rend::pyhost {
// Function-pointer types for GetProcAddress consumers.
using StartFn = bool (*)(rend::renderer::MessageQueue*, const char* const*, int);
using StopFn = void (*)(void);
using RunningFn = bool (*)(void);
} // namespace rend::pyhost

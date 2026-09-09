#include "rend/pyhost/pyhost.h"

#include "rend/core/log.h"
#include "rend/core/profile.h"
#include "rend/renderer/message_queue.h"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using namespace rend;

namespace {

// One host per process: the embedded module functions need a place to
// find the queue, and CPython itself is a process-global anyway (a second
// interpreter would fight over the GIL and extension state).
struct HostState {
    renderer::MessageQueue* queue = nullptr;
    std::optional<renderer::Sender> sender;         // script-thread-only
    std::optional<renderer::EventReceiver> receiver; // script-thread-only
    std::vector<renderer::Event> eventCache; // polled but unconsumed events
    std::atomic<bool> quit{false};
    std::thread thread;
    // Guards interpreterAlive against the Stop-side PyErr_SetInterrupt:
    // the flag drops (under the mutex) before finalization begins, so the
    // interrupt is never delivered into a dying interpreter.
    std::mutex interpMutex;
    bool interpreterAlive = false;
};

HostState g_host;

py::dict eventToDict(const renderer::Event& event) {
    py::dict out;
    switch (event.type) {
    case renderer::Event::Type::ModelReady:
        out["type"] = "model_ready";
        out["handle"] = event.ready.handle;
        out["ok"] = event.ready.ok;
        out["millis"] = event.ready.millis;
        out["error"] = std::string(event.ready.error);
        break;
    }
    return out;
}

// Pulls freshly broadcast events into the host cache (so a wait for one
// handle never swallows events a later poll should still see).
void refillEventCache() {
    auto fresh = g_host.receiver->poll();
    g_host.eventCache.insert(g_host.eventCache.end(), fresh.begin(), fresh.end());
}

void pushCommand(const renderer::Command& command) {
    g_host.sender->push(command);
    g_host.sender->flush(); // one command per commit: scripts are sparse
}

} // namespace

// The `rend` module scripts import. Registered via inittab at DLL load,
// so it exists for any interpreter this process starts afterwards.
PYBIND11_EMBEDDED_MODULE(rend, m) {
    m.doc() = "Renderer message-queue bindings (embedded script host)";

    m.def(
        "load_model",
        [](const std::string& path, std::array<float, 3> position, float yaw, float scale) {
            renderer::Command cmd;
            cmd.type = renderer::Command::Type::LoadModel;
            cmd.load.handle = g_host.queue->allocateHandle();
            if (path.size() >= sizeof(cmd.load.path)) {
                throw py::value_error("model path too long (max 511 bytes)");
            }
            std::snprintf(cmd.load.path, sizeof(cmd.load.path), "%s", path.c_str());
            cmd.load.position[0] = position[0];
            cmd.load.position[1] = position[1];
            cmd.load.position[2] = position[2];
            cmd.load.yawDegrees = yaw;
            cmd.load.scale = scale;
            pushCommand(cmd);
            return cmd.load.handle;
        },
        py::arg("path"), py::arg("position") = std::array<float, 3>{0.0f, 0.0f, 0.0f},
        py::arg("yaw") = 0.0f, py::arg("scale") = 1.0f,
        "Queue a model load; returns its handle. The model exists only "
        "once a model_ready event for the handle reports ok.");

    m.def(
        "set_transform",
        [](renderer::ModelHandle handle, std::array<float, 3> position, float yaw, float scale) {
            renderer::Command cmd;
            cmd.type = renderer::Command::Type::SetTransform;
            cmd.transform.handle = handle;
            cmd.transform.position[0] = position[0];
            cmd.transform.position[1] = position[1];
            cmd.transform.position[2] = position[2];
            cmd.transform.yawDegrees = yaw;
            cmd.transform.scale = scale;
            pushCommand(cmd);
        },
        py::arg("handle"), py::arg("position"), py::arg("yaw") = 0.0f, py::arg("scale") = 1.0f,
        "Move/rotate/scale a loaded model.");

    m.def(
        "unload_model",
        [](renderer::ModelHandle handle) {
            renderer::Command cmd;
            cmd.type = renderer::Command::Type::UnloadModel;
            cmd.unload.handle = handle;
            pushCommand(cmd);
        },
        py::arg("handle"), "Remove a loaded model from the scene.");

    m.def(
        "set_sun",
        [](std::array<float, 3> direction, std::array<float, 3> color, float intensity) {
            renderer::Command cmd;
            cmd.type = renderer::Command::Type::SetSun;
            for (int i = 0; i < 3; ++i) {
                cmd.sun.direction[i] = direction[i];
                cmd.sun.color[i] = color[i];
            }
            cmd.sun.intensity = intensity;
            pushCommand(cmd);
        },
        py::arg("direction"), py::arg("color") = std::array<float, 3>{1.0f, 1.0f, 1.0f},
        py::arg("intensity") = 1.0f,
        "Set the directional sun: direction points from the light toward "
        "the scene; intensity 0 turns the sun off for the frame.");

    m.def(
        "set_sky_color",
        [](std::array<float, 3> color) {
            renderer::Command cmd;
            cmd.type = renderer::Command::Type::SetSkyColor;
            for (int i = 0; i < 3; ++i) {
                cmd.sky.color[i] = color[i];
            }
            pushCommand(cmd);
        },
        py::arg("color"),
        "Set the background/sky color the frame paints behind the scene.");

    m.def(
        "set_time_of_day",
        [](float t) {
            renderer::Command cmd;
            cmd.type = renderer::Command::Type::SetTimeOfDay;
            cmd.timeOfDay.t = t;
            pushCommand(cmd);
        },
        py::arg("t"),
        "Drive the procedural skybox: day phase in [0,1) (0 sunrise, 0.25 "
        "noon, 0.5 sunset). Negative disables it (flat sky color).");

    m.def(
        "set_ambient",
        [](std::array<float, 3> color) {
            renderer::Command cmd;
            cmd.type = renderer::Command::Type::SetAmbient;
            for (int i = 0; i < 3; ++i) {
                cmd.ambient.color[i] = color[i];
            }
            pushCommand(cmd);
        },
        py::arg("color"),
        "Set the hemispherical ambient tint (default 0.30 0.32 0.36).");

    m.def(
        "set_point_light",
        [](std::uint32_t index, std::array<float, 3> position, std::array<float, 3> color,
           float intensity, float radius, bool castsShadows) {
            renderer::Command cmd;
            cmd.type = renderer::Command::Type::SetPointLight;
            cmd.pointLight.index = index;
            for (int i = 0; i < 3; ++i) {
                cmd.pointLight.position[i] = position[i];
                cmd.pointLight.color[i] = color[i];
            }
            cmd.pointLight.intensity = intensity;
            cmd.pointLight.radius = radius;
            cmd.pointLight.castsShadows = castsShadows ? 1u : 0u;
            pushCommand(cmd);
        },
        py::arg("index"), py::arg("position"),
        py::arg("color") = std::array<float, 3>{1.0f, 1.0f, 1.0f}, py::arg("intensity") = 1.0f,
        py::arg("radius") = 10.0f, py::arg("casts_shadows") = false,
        "Set one dynamic point light slot (16 slots); intensity 0 turns "
        "the slot off. casts_shadows applies to traced shadow rays only — "
        "baked shadow cubes belong to scene-XML lights.");

    m.def(
        "set_point_light_scale",
        [](float scale) {
            renderer::Command cmd;
            cmd.type = renderer::Command::Type::SetPointLightScale;
            cmd.pointLightScale.scale = scale;
            pushCommand(cmd);
        },
        py::arg("scale"),
        "Scale every point light's authored intensity (0 = all off) — the "
        "one-call fade for scene-XML lamps.");

    m.def(
        "poll_events",
        [] {
            refillEventCache();
            py::list out;
            for (const renderer::Event& event : g_host.eventCache) {
                out.append(eventToDict(event));
            }
            g_host.eventCache.clear();
            return out;
        },
        "Non-blocking: every event delivered since the last poll/wait, as "
        "dicts.");

    m.def(
        "wait_model_ready",
        [](renderer::ModelHandle handle, double timeout) -> py::object {
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout);
            for (;;) {
                refillEventCache();
                for (auto it = g_host.eventCache.begin(); it != g_host.eventCache.end(); ++it) {
                    if (it->type == renderer::Event::Type::ModelReady &&
                        it->ready.handle == handle) {
                        const renderer::Event event = *it;
                        g_host.eventCache.erase(it);
                        return eventToDict(event);
                    }
                }
                if (g_host.quit.load() || std::chrono::steady_clock::now() >= deadline) {
                    return py::none();
                }
                py::gil_scoped_release release;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        },
        py::arg("handle"), py::arg("timeout") = 30.0,
        "Block (Python-side only) until the handle's model_ready event "
        "arrives; returns its dict, or None on timeout/shutdown. Other "
        "events stay queued for poll_events().");

    m.def(
        "should_quit", [] { return g_host.quit.load(); },
        "True once the app asked scripts to stop; long-running loops must "
        "poll this.");

    m.def(
        "log", [](const std::string& message) { log::info("[py] {}", message); },
        py::arg("message"), "Write into the renderer log.");
}

namespace {

std::wstring pythonHome() {
    if (const wchar_t* env = _wgetenv(L"PYTHONHOME"); env && *env) {
        return env;
    }
    const std::string baked = REND_PYHOST_PYTHON_HOME;
    return std::wstring(baked.begin(), baked.end()); // build path: ASCII
}

void scriptThreadMain(std::vector<std::string> scripts) {
    REND_PROFILE_THREAD("python");
    PyConfig config;
    PyConfig_InitIsolatedConfig(&config); // no user site/env leakage
    // Isolated config skips signal handlers, but Stop's PyErr_SetInterrupt
    // needs them to deliver a clean KeyboardInterrupt into blocking sleeps
    // (without them Windows raises "Signal 2 ignored due to race
    // condition" OSErrors instead).
    config.install_signal_handlers = 1;
    const std::wstring home = pythonHome();
    PyStatus status = PyConfig_SetString(&config, &config.home, home.c_str());
    if (PyStatus_Exception(status)) {
        PyConfig_Clear(&config);
        log::error("[py] interpreter config failed: {}",
                   status.err_msg ? status.err_msg : "unknown");
        return;
    }
    try {
        py::scoped_interpreter interpreter{&config};
        {
            std::lock_guard lock(g_host.interpMutex);
            g_host.interpreterAlive = true;
        }
        log::info("[py] interpreter {} up, running {} script(s)", PY_VERSION, scripts.size());
        for (const std::string& script : scripts) {
            if (g_host.quit.load()) {
                break;
            }
            try {
                py::dict globals;
                globals["__builtins__"] = py::module_::import("builtins");
                globals["__name__"] = "__main__";
                globals["__file__"] = script;
                py::eval_file(script, globals);
                log::info("[py] '{}' completed", script);
            } catch (py::error_already_set& error) {
                if (error.matches(PyExc_KeyboardInterrupt) || g_host.quit.load()) {
                    log::info("[py] '{}' interrupted (shutdown)", script);
                } else {
                    log::error("[py] '{}' raised:\n{}", script, error.what());
                }
            }
        }
        {
            std::lock_guard lock(g_host.interpMutex);
            g_host.interpreterAlive = false;
        }
    } catch (const std::exception& error) {
        std::lock_guard lock(g_host.interpMutex);
        g_host.interpreterAlive = false;
        log::error("[py] interpreter failed: {}", error.what());
    }
}

} // namespace

extern "C" {

bool rendPyHostStart(rend::renderer::MessageQueue* queue, const char* const* scriptPaths,
                     int scriptCount) {
    if (!queue || scriptCount <= 0 || !scriptPaths) {
        return false;
    }
    if (g_host.thread.joinable()) {
        log::warn("[py] host already running; Start ignored");
        return false;
    }
    g_host.queue = queue;
    g_host.sender.emplace(queue->createSender());
    // Receiver exists before any script can issue a command: no completion
    // event can ever be missed.
    g_host.receiver.emplace(queue->createEventReceiver());
    g_host.eventCache.clear();
    g_host.quit.store(false);
    std::vector<std::string> scripts(scriptPaths, scriptPaths + scriptCount);
    g_host.thread = std::thread(scriptThreadMain, std::move(scripts));
    return true;
}

void rendPyHostStop(void) {
    if (!g_host.thread.joinable()) {
        return;
    }
    g_host.quit.store(true);
    {
        // KeyboardInterrupt breaks time.sleep()-style waits; guarded so it
        // can never land in an interpreter that is already finalizing.
        std::lock_guard lock(g_host.interpMutex);
        if (g_host.interpreterAlive) {
            PyErr_SetInterrupt();
        }
    }
    g_host.thread.join();
    g_host.receiver.reset();
    g_host.sender.reset();
    g_host.queue = nullptr;
}

bool rendPyHostRunning(void) {
    std::lock_guard lock(g_host.interpMutex);
    return g_host.interpreterAlive;
}
}

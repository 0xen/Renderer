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
#include <map>
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
// Mirrored graphics-settings state: the viewer broadcasts every slot's
// SettingState at startup and on change, so the get_setting* bindings
// answer synchronously from this cache — no request/response round trip
// over the queue. Touched only with the GIL held (like eventCache).
struct SettingMirror {
    std::string active;
    std::string overriddenBy;
    std::vector<std::string> options;
};

// Mirrored animation state, one entry per animated model name: what is
// playing and the full clip menu, refreshed by every AnimationState
// broadcast. Same reasoning as SettingMirror — synchronous getters.
struct AnimationMirror {
    std::string clip;
    std::vector<std::string> clips;
    float duration = 0.0f;
    float time = 0.0f; // as of the last broadcast, not a live clock
    float speed = 1.0f;
    bool paused = false;
};

struct HostState {
    renderer::MessageQueue* queue = nullptr;
    std::optional<renderer::Sender> sender;         // script-thread-only
    std::optional<renderer::EventReceiver> receiver; // script-thread-only
    std::vector<renderer::Event> eventCache; // polled but unconsumed events
    std::map<std::string, SettingMirror> settings;
    std::map<std::string, AnimationMirror> animations;
    // Latest broadcast camera pose (same mirror idea as settings): the
    // viewer sends CameraState whenever the pose changes, get_camera
    // answers from here. Empty until the first broadcast lands.
    std::optional<renderer::CameraStateEvent> cameraMirror;
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
    case renderer::Event::Type::SettingRejected:
        out["type"] = "setting_rejected";
        out["name"] = std::string(event.settingRejected.name);
        out["value"] = std::string(event.settingRejected.value);
        out["reason"] = std::string(event.settingRejected.reason);
        break;
    case renderer::Event::Type::SettingState:
    case renderer::Event::Type::CameraState:
    case renderer::Event::Type::AnimationState:
        break; // absorbed into their mirrors, never surfaced raw
    }
    return out;
}

// Pulls freshly broadcast events into the host cache (so a wait for one
// handle never swallows events a later poll should still see). Settings
// broadcasts fold into the mirror instead of the cache — they are state
// sync, not completions a script waits on.
void refillEventCache() {
    for (const renderer::Event& event : g_host.receiver->poll()) {
        if (event.type == renderer::Event::Type::SettingState) {
            SettingMirror& mirror = g_host.settings[event.settingState.name];
            mirror.active = event.settingState.active;
            mirror.overriddenBy = event.settingState.overriddenBy;
            mirror.options.clear();
            for (std::uint32_t i = 0;
                 i < event.settingState.optionCount && i < renderer::kSettingMaxOptions; ++i) {
                mirror.options.emplace_back(event.settingState.options[i]);
            }
        } else if (event.type == renderer::Event::Type::CameraState) {
            g_host.cameraMirror = event.cameraState;
        } else if (event.type == renderer::Event::Type::AnimationState) {
            AnimationMirror& mirror = g_host.animations[event.animationState.model];
            mirror.clip = event.animationState.clip;
            mirror.duration = event.animationState.duration;
            mirror.time = event.animationState.time;
            mirror.speed = event.animationState.speed;
            mirror.paused = event.animationState.paused != 0;
            mirror.clips.clear();
            for (std::uint32_t i = 0;
                 i < event.animationState.clipCount && i < renderer::kAnimationMaxClips; ++i) {
                mirror.clips.emplace_back(event.animationState.clips[i]);
            }
        } else {
            g_host.eventCache.push_back(event);
        }
    }
}

const SettingMirror* findSetting(const std::string& name) {
    refillEventCache();
    const auto it = g_host.settings.find(name);
    return it != g_host.settings.end() ? &it->second : nullptr;
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
        [](const std::string& path, std::array<float, 3> position, float yaw, float scale,
           bool reflective) {
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
            cmd.load.reflective = reflective ? 1u : 0u;
            pushCommand(cmd);
            return cmd.load.handle;
        },
        py::arg("path"), py::arg("position") = std::array<float, 3>{0.0f, 0.0f, 0.0f},
        py::arg("yaw") = 0.0f, py::arg("scale") = 1.0f, py::arg("reflective") = false,
        "Queue a model load; returns its handle. The model exists only "
        "once a model_ready event for the handle reports ok. reflective "
        "tags the model for RT/probe reflections (the scene-XML "
        "reflective=\"true\" twin); the first load of a path decides for "
        "every instance of it.");

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
        "set_camera",
        [](std::array<float, 3> position, std::array<float, 3> target) {
            renderer::Command cmd;
            cmd.type = renderer::Command::Type::SetCamera;
            for (int i = 0; i < 3; ++i) {
                cmd.camera.position[i] = position[i];
                cmd.camera.target[i] = target[i];
            }
            pushCommand(cmd);
        },
        py::arg("position"), py::arg("target"),
        "Place the free-fly camera at position looking at target. Cancels "
        "a scene fly-in; the user's mouselook/WASD continue from the last "
        "scripted pose once the script stops sending.");

    m.def(
        "get_camera",
        []() -> py::object {
            refillEventCache();
            if (!g_host.cameraMirror) {
                return py::none();
            }
            const renderer::CameraStateEvent& cam = *g_host.cameraMirror;
            py::dict out;
            out["position"] =
                py::make_tuple(cam.position[0], cam.position[1], cam.position[2]);
            out["yaw"] = cam.yaw;
            out["pitch"] = cam.pitch;
            out["fov_degrees"] = cam.fovDegrees;
            return out;
        },
        "The free-fly camera's latest broadcast pose as {position, yaw, "
        "pitch, fov_degrees} (yaw/pitch radians: yaw 0 looks down -Z, "
        "positive turns toward +X; pitch positive looks up). None until "
        "the first frame's broadcast arrives. Updates lag the frame loop "
        "by up to a frame — poll, don't assume instant echo of "
        "set_camera.");

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
        "set_setting",
        [](const std::string& name, const std::string& value) {
            renderer::Command cmd;
            cmd.type = renderer::Command::Type::SetSetting;
            if (name.size() >= sizeof(cmd.setting.name)) {
                throw py::value_error("setting name too long");
            }
            if (value.size() >= sizeof(cmd.setting.value)) {
                throw py::value_error("setting value too long");
            }
            std::snprintf(cmd.setting.name, sizeof(cmd.setting.name), "%s", name.c_str());
            std::snprintf(cmd.setting.value, sizeof(cmd.setting.value), "%s", value.c_str());
            pushCommand(cmd);
        },
        py::arg("name"), py::arg("value"),
        "Request a graphics setting change (e.g. set_setting('shadows', "
        "'raytraced')). Asynchronous: an accepted change shows up in the "
        "getters next frame; a refused one (unknown name, unlisted option, "
        "overridden slot) arrives as a 'setting_rejected' event.");

    m.def(
        "list_settings",
        [] {
            refillEventCache();
            py::list out;
            for (const auto& [name, mirror] : g_host.settings) {
                out.append(name);
            }
            return out;
        },
        "Names of every graphics setting the active renderer exposes.");

    m.def(
        "get_setting",
        [](const std::string& name) -> py::object {
            if (const SettingMirror* mirror = findSetting(name)) {
                return py::str(mirror->active);
            }
            return py::none();
        },
        py::arg("name"),
        "The setting's effective value: its option token, a number as a "
        "string, or 'override' when another setting supersedes it (see "
        "get_override_source). None for unknown names.");

    m.def(
        "get_setting_options",
        [](const std::string& name) {
            py::list out;
            if (const SettingMirror* mirror = findSetting(name)) {
                for (const std::string& option : mirror->options) {
                    out.append(option);
                }
            }
            return out;
        },
        py::arg("name"),
        "Option tokens selectable RIGHT NOW. Empty while the setting is "
        "overridden, for continuous (numeric) settings, and for unknown "
        "names.");

    m.def(
        "get_override_source",
        [](const std::string& name) -> py::object {
            if (const SettingMirror* mirror = findSetting(name);
                mirror && !mirror->overriddenBy.empty()) {
                return py::str(mirror->overriddenBy);
            }
            return py::none();
        },
        py::arg("name"),
        "Name of the setting overriding this one (e.g. 'primary' while "
        "traced primary rays subsume the shadow choice); None when not "
        "overridden.");

    // Every animation binding funnels through here: one command type with a
    // field mask, so changing the speed never re-triggers the clip.
    auto pushAnimation = [](const std::string& model, std::uint32_t fields,
                            const std::string& clip, float blend, float speed, float time,
                            bool paused) {
        if (clip.size() >= renderer::kAnimationNameChars ||
            model.size() >= renderer::kAnimationNameChars) {
            throw py::value_error("model/clip name too long");
        }
        renderer::Command cmd;
        cmd.type = renderer::Command::Type::SetAnimation;
        cmd.animation = {};
        std::snprintf(cmd.animation.model, sizeof(cmd.animation.model), "%s", model.c_str());
        std::snprintf(cmd.animation.clip, sizeof(cmd.animation.clip), "%s", clip.c_str());
        cmd.animation.blendSeconds = blend;
        cmd.animation.speed = speed;
        cmd.animation.time = time;
        cmd.animation.paused = paused ? 1u : 0u;
        cmd.animation.fields = fields;
        pushCommand(cmd);
    };

    m.def(
        "set_animation",
        [pushAnimation](const std::string& clip, const std::string& model, float blend) {
            pushAnimation(model, renderer::kAnimationFieldClip, clip, blend, 1.0f, 0.0f,
                          false);
        },
        py::arg("clip"), py::arg("model") = std::string(), py::arg("blend") = 0.25f,
        "Switch an animated model to the named clip, cross-fading out of "
        "the running one over `blend` seconds (0 = snap). `model` is a "
        "scene-XML <Model name>; the default empty string targets EVERY "
        "animated model. Async like every command: a clip the model does "
        "not have is logged viewer-side and ignored, so confirm with "
        "get_animation() rather than assuming.");

    m.def(
        "set_animation_speed",
        [pushAnimation](float speed, const std::string& model) {
            pushAnimation(model, renderer::kAnimationFieldSpeed, {}, 0.0f, speed, 0.0f, false);
        },
        py::arg("speed"), py::arg("model") = std::string(),
        "Playback time scale: 1.0 normal, 0.5 half speed, 0 freezes the "
        "clip where it stands, negative plays it backwards (times wrap at "
        "both ends). Leaves the clip and the pause flag alone.");

    m.def(
        "pause_animation",
        [pushAnimation](bool paused, const std::string& model) {
            pushAnimation(model, renderer::kAnimationFieldPaused, {}, 0.0f, 1.0f, 0.0f,
                          paused);
        },
        py::arg("paused") = true, py::arg("model") = std::string(),
        "Hold the clip (and any running cross-fade) at its current time. "
        "The pose is still re-evaluated every frame, so a seek or a clip "
        "switch while paused shows immediately. Unlike speed=0 this "
        "remembers the speed to resume at.");

    m.def(
        "set_animation_time",
        [pushAnimation](float seconds, const std::string& model) {
            pushAnimation(model, renderer::kAnimationFieldTime, {}, 0.0f, 1.0f, seconds,
                          false);
        },
        py::arg("seconds"), py::arg("model") = std::string(),
        "Seek the playing clip, wrapping into [0, duration). Combine with "
        "pause_animation() to step frame by frame; the resulting time "
        "comes back in get_animation_state().");

    m.def(
        "list_animations",
        [](const std::string& model) {
            refillEventCache();
            py::dict out;
            for (const auto& [name, mirror] : g_host.animations) {
                if (!model.empty() && name != model) {
                    continue;
                }
                py::list clips;
                for (const std::string& clip : mirror.clips) {
                    clips.append(clip);
                }
                out[py::str(name)] = clips;
            }
            return out;
        },
        py::arg("model") = std::string(),
        "{model name: [clip names]} for every animated model in the scene "
        "(one entry when `model` names one). Mirrored state, so it answers "
        "instantly — but it is EMPTY until the viewer's first broadcast "
        "lands, exactly like the settings mirror.");

    m.def(
        "get_animation",
        [](const std::string& model) -> py::object {
            refillEventCache();
            if (model.empty()) {
                if (g_host.animations.size() == 1) {
                    return py::str(g_host.animations.begin()->second.clip);
                }
                return py::none();
            }
            const auto it = g_host.animations.find(model);
            return it != g_host.animations.end() ? py::object(py::str(it->second.clip))
                                                 : py::none();
        },
        py::arg("model") = std::string(),
        "The clip a model is playing now, from the mirror. With no model "
        "name it answers only when the scene has exactly ONE animated "
        "model; otherwise None (name the model).");

    m.def(
        "get_animation_state",
        [](const std::string& model) -> py::object {
            refillEventCache();
            auto it = g_host.animations.end();
            if (model.empty()) {
                if (g_host.animations.size() == 1) {
                    it = g_host.animations.begin();
                }
            } else {
                it = g_host.animations.find(model);
            }
            if (it == g_host.animations.end()) {
                return py::none();
            }
            py::list clips;
            for (const std::string& clip : it->second.clips) {
                clips.append(clip);
            }
            py::dict out;
            out["model"] = it->first;
            out["clip"] = it->second.clip;
            out["clips"] = clips;
            out["duration"] = it->second.duration;
            out["time"] = it->second.time;
            out["speed"] = it->second.speed;
            out["paused"] = it->second.paused;
            return out;
        },
        py::arg("model") = std::string(),
        "Full mirrored playback state as a dict: model, clip, clips, "
        "duration, time, speed, paused. NOTE `time` is the clip time at "
        "the last state change (state broadcasts on change, not per "
        "frame), so it is exact after a seek or switch and stale while a "
        "clip free-runs. None when the name is unknown, or when the name "
        "is omitted and the scene has more than one animated model.");

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
    g_host.settings.clear();
    g_host.cameraMirror.reset();
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

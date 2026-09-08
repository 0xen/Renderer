#pragma once

#include <cstdint>

namespace rend::renderer {

// Stable identity a producer invents (MessageQueue::allocateHandle) and
// uses to refer to one loaded model across messages and events.
using ModelHandle = std::uint32_t;
inline constexpr ModelHandle kInvalidModel = 0;

// Command payloads are plain PODs — fixed-size strings, no pointers, no
// STL — so the same schema can later cross a C / Python / IPC boundary
// unchanged. The consumer (today: the viewer's frame loop; later: the
// renderer layer proper) resolves paths through the app-side asset stack;
// the engine itself never opens files (docs/ARCHITECTURE.md).

struct LoadModelCmd {
    ModelHandle handle = kInvalidModel;
    char path[512] = {};   // model file, UTF-8
    float position[3] = {};
    float yawDegrees = 0.0f;
    float scale = 1.0f;
};

struct SetTransformCmd {
    ModelHandle handle = kInvalidModel;
    float position[3] = {};
    float yawDegrees = 0.0f;
    float scale = 1.0f;
};

struct UnloadModelCmd {
    ModelHandle handle = kInvalidModel;
};

// Lighting state (the script-driven day/night cycle). Each command
// replaces the named viewer-side state wholesale; the frame loop copies
// it into the per-slot light buffer, so per-frame updates never stall.

struct SetSunCmd {
    float direction[3] = {0.0f, -1.0f, 0.0f}; // from the light toward the scene
    float color[3] = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
};

struct SetSkyColorCmd {
    float color[3] = {};
};

struct SetAmbientCmd {
    float color[3] = {};
};

// One dynamic point light slot (no shadows); intensity 0 turns it off.
// The consumer bounds the slot count (16 today) and ignores out-of-range
// indices with a warning.
struct SetPointLightCmd {
    std::uint32_t index = 0;
    float position[3] = {};
    float color[3] = {1.0f, 1.0f, 1.0f};
    float intensity = 0.0f;
    float radius = 10.0f;
};

struct Command {
    enum class Type : std::uint32_t {
        LoadModel,
        SetTransform,
        UnloadModel,
        SetSun,
        SetSkyColor,
        SetAmbient,
        SetPointLight,
    };
    Type type = Type::LoadModel;
    union {
        LoadModelCmd load;
        SetTransformCmd transform;
        UnloadModelCmd unload;
        SetSunCmd sun;
        SetSkyColorCmd sky;
        SetAmbientCmd ambient;
        SetPointLightCmd pointLight;
    };
    Command() : load{} {}
};

// Events flow back from the consumer: a load completion (or failure) the
// producer polls for. Loads are asynchronous by contract even while the
// implementation is synchronous — producers must not assume the model
// exists until ModelReady arrives.
struct ModelReadyEvent {
    ModelHandle handle = kInvalidModel;
    bool ok = false;
    float millis = 0.0f; // wall time the load took, for the test harness
    char error[160] = {};
};

struct Event {
    enum class Type : std::uint32_t { ModelReady };
    Type type = Type::ModelReady;
    union {
        ModelReadyEvent ready;
    };
    Event() : ready{} {}
};

} // namespace rend::renderer

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

struct Command {
    enum class Type : std::uint32_t { LoadModel, SetTransform, UnloadModel };
    Type type = Type::LoadModel;
    union {
        LoadModelCmd load;
        SetTransformCmd transform;
        UnloadModelCmd unload;
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

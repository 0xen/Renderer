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
    // Semantic reflective tag, the runtime twin of the scene-XML
    // reflective="true" attribute. Material state is per RESOURCE (shared
    // by every instance of the path), so the first load of a path decides;
    // later loads with a different value warn and keep the first.
    std::uint32_t reflective = 0;
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

// Skybox day phase in [0,1) (0 sunrise, 0.25 noon, 0.5 sunset): enables
// the sky pass's procedural cube and drives its palette. Negative
// disables the skybox and falls back to the flat SetSkyColor color.
struct SetTimeOfDayCmd {
    float t = -1.0f;
};

struct SetAmbientCmd {
    float color[3] = {};
};

// One dynamic point light slot; intensity 0 turns it off. The consumer
// bounds the slot count (16 today) and ignores out-of-range indices with
// a warning. Replacing a slot drops any baked shadow cube it had (a
// repositioned light can't reuse it); castsShadows still gates traced
// shadow rays.
struct SetPointLightCmd {
    std::uint32_t index = 0;
    float position[3] = {};
    float color[3] = {1.0f, 1.0f, 1.0f};
    float intensity = 0.0f;
    float radius = 10.0f;
    std::uint32_t castsShadows = 0;
};

// Global multiplier over every point light's authored intensity — the
// one-call way for a script to fade the scene's lamps in and out.
struct SetPointLightScaleCmd {
    float scale = 1.0f;
};

// Place the free-fly camera: position plus a look-at target the consumer
// converts to yaw/pitch, so user mouselook continues naturally from the
// scripted pose the moment commands stop arriving. Cancels any scene
// fly-in in progress. Scripts animate by sending one per step.
struct SetCameraCmd {
    float position[3] = {};
    float target[3] = {0.0f, 0.0f, -1.0f};
};

// Graphics-settings slot names and option tokens are short strings (the
// registry in settings.h validates them); a rejected set comes back as a
// SettingRejected event, an accepted one as the slot's SettingState.
inline constexpr std::uint32_t kSettingNameChars = 32;
inline constexpr std::uint32_t kSettingOptionChars = 32;
inline constexpr std::uint32_t kSettingMaxOptions = 8;

struct SetSettingCmd {
    char name[kSettingNameChars] = {};    // e.g. "shadows"
    char value[kSettingOptionChars] = {}; // e.g. "raytraced", "on", "0.08"
};

// Animation clip selection. Model and clip are named rather than indexed
// because a producer sees names, not the viewer's load order; an empty
// model targets every animated model in the scene. A clip the model does
// not have is logged and ignored (the mirrored AnimationState simply does
// not change, which is how a script detects the miss).
inline constexpr std::uint32_t kAnimationNameChars = 48;
inline constexpr std::uint32_t kAnimationMaxClips = 16;

// Which members of SetAnimationCmd the producer actually set. One command
// carries the whole playback vocabulary, and the mask is what keeps
// "change the speed" from also re-triggering the clip.
enum AnimationField : std::uint32_t {
    kAnimationFieldClip = 1u << 0,
    kAnimationFieldSpeed = 1u << 1,
    kAnimationFieldPaused = 1u << 2,
    kAnimationFieldTime = 1u << 3,
    kAnimationFieldLoop = 1u << 4,
};

struct SetAnimationCmd {
    char model[kAnimationNameChars] = {}; // empty = every animated model
    char clip[kAnimationNameChars] = {};
    float blendSeconds = 0.25f; // cross-fade out of the running clip
    float speed = 1.0f;         // time scale; 0 freezes, negative plays backwards
    float time = 0.0f;          // seek target in seconds, wrapped into the clip
    std::uint32_t paused = 0;
    std::uint32_t loop = 1; // 0 = play once and hold the last pose
    std::uint32_t fields = 0; // AnimationField bits
};

struct Command {
    enum class Type : std::uint32_t {
        LoadModel,
        SetTransform,
        UnloadModel,
        SetSun,
        SetSkyColor,
        SetTimeOfDay,
        SetAmbient,
        SetPointLight,
        SetPointLightScale,
        SetCamera,
        SetSetting,
        SetAnimation,
    };
    Type type = Type::LoadModel;
    union {
        LoadModelCmd load;
        SetTransformCmd transform;
        UnloadModelCmd unload;
        SetSunCmd sun;
        SetSkyColorCmd sky;
        SetTimeOfDayCmd timeOfDay;
        SetAmbientCmd ambient;
        SetPointLightCmd pointLight;
        SetPointLightScaleCmd pointLightScale;
        SetCameraCmd camera;
        SetSettingCmd setting;
        SetAnimationCmd animation;
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

// The free-fly camera's pose, broadcast whenever it changes (user input,
// scene fly-in, SetCamera). Consumers mirror the latest one so a
// synchronous get_camera can answer without a round trip; yaw/pitch are
// radians in the FlyCamera convention (yaw 0 looks down -Z, positive
// turns toward +X; pitch positive looks up).
struct CameraStateEvent {
    float position[3] = {};
    float yaw = 0.0f;
    float pitch = 0.0f;
    float fovDegrees = 60.0f;
};

// One graphics-setting slot's full public state, broadcast at startup and
// whenever it changes (any producer's set, a UI flip, a capability
// arriving). `active` is the EFFECTIVE answer: the selected option token,
// a formatted number for continuous slots, or the literal "override" when
// `overriddenBy` names the slot that superseded this one — in which case
// optionCount is 0 (nothing is selectable until the override lifts).
struct SettingStateEvent {
    char name[kSettingNameChars] = {};
    char active[kSettingOptionChars] = {};
    char overriddenBy[kSettingNameChars] = {};
    std::uint32_t optionCount = 0;
    char options[kSettingMaxOptions][kSettingOptionChars] = {};
};

// A SetSetting the consumer refused (unknown slot, unlisted option,
// overridden slot). The producer's only failure signal — the accepted
// case answers with SettingState instead.
struct SettingRejectedEvent {
    char name[kSettingNameChars] = {};
    char value[kSettingOptionChars] = {};
    char reason[160] = {};
};

// One animated model's clip state, broadcast at load and on every switch
// (the same mirror idea as SettingState): `clip` is what is playing now,
// `clips` the full menu. Models with more than kAnimationMaxClips clips
// report the first kAnimationMaxClips — clipCount is the reported count,
// not necessarily the model's total.
struct AnimationStateEvent {
    char model[kAnimationNameChars] = {};
    char clip[kAnimationNameChars] = {};
    float duration = 0.0f; // of the playing clip, seconds
    // Playback time AT THE MOMENT OF THE BROADCAST. State broadcasts only
    // on change, so this is exact right after a switch/seek and goes stale
    // while the clip free-runs — it is a seek acknowledgement, not a clock.
    float time = 0.0f;
    float speed = 1.0f;
    std::uint32_t paused = 0;
    std::uint32_t loop = 1; // of the playing clip
    std::uint32_t clipCount = 0;
    char clips[kAnimationMaxClips][kAnimationNameChars] = {};
};

struct Event {
    enum class Type : std::uint32_t {
        ModelReady,
        SettingState,
        SettingRejected,
        CameraState,
        AnimationState,
    };
    Type type = Type::ModelReady;
    union {
        ModelReadyEvent ready;
        SettingStateEvent settingState;
        SettingRejectedEvent settingRejected;
        CameraStateEvent cameraState;
        AnimationStateEvent animationState;
    };
    Event() : ready{} {}
};

} // namespace rend::renderer

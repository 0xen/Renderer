#pragma once

#include "rend/core/result.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace rend::renderer {

// The active() answer for a setting another setting has superseded (e.g.
// traced primary rays subsume the shadow/reflection technique choice).
// Overridden slots also report an EMPTY option list: there is nothing to
// select until the overriding setting releases them.
inline constexpr const char* kSettingOverride = "override";

// Single authority for the renderer's user-facing graphics settings.
// Every mutation path — ImGui, Python commands, CLI-seeded defaults —
// goes through set()/setFloat(), so the apply callbacks are the ONE place
// that touches renderer state (and the one place that knows whether a
// setting is live data or baked into static recordings). Queries feed the
// UI and the SettingState events the message queue broadcasts.
//
// Not thread-safe by design: registration and every call happen on the
// frame-loop thread (Python mutations arrive as drained commands there).
class SettingsRegistry {
public:
    // A choice among string option tokens (e.g. "raster"/"raytraced").
    // apply() commits the value to renderer state and returns false to
    // veto (leaving the previous value active). Options may legally be a
    // single entry — the setting is visible but has only one answer.
    void addChoice(std::string name, std::vector<std::string> options, std::string active,
                   std::function<bool(const std::string&)> apply);

    // A continuous value (e.g. fog density). Choice-style queries report
    // an empty option list; active() formats the value.
    void addFloat(std::string name, float minValue, float maxValue, float value,
                  std::function<bool(float)> apply);

    // Capability changes after registration (a probe finishing its bake
    // adds "probe" to the reflection options). Keeps the active value if
    // still listed; otherwise snaps to the first option (without apply —
    // the caller changed the world, it owns the consequences).
    void setOptions(const std::string& name, std::vector<std::string> options);

    // Mark a slot superseded by `source` (another slot's name) or release
    // it. Overridden slots reject set(), report active() = "override" and
    // an empty option list.
    void setOverride(const std::string& name, const std::string& source);
    void clearOverride(const std::string& name);

    // ---- Queries (the get-* surface the UI / events are built from) ----
    std::vector<std::string> names() const;
    bool exists(const std::string& name) const;
    // Empty when the slot is overridden, continuous, or unknown.
    std::vector<std::string> options(const std::string& name) const;
    // The selected option token; kSettingOverride when overridden; "" when
    // unknown. Float slots format their value ("0.035").
    std::string active(const std::string& name) const;
    // The overriding slot's name; "" when not overridden.
    std::string overrideSource(const std::string& name) const;
    // Float access for slider-style UI. floatValue on a choice slot is 0.
    bool isFloat(const std::string& name) const;
    float floatValue(const std::string& name) const;
    float floatMin(const std::string& name) const;
    float floatMax(const std::string& name) const;

    // ---- Mutation (validates, applies, records the change) ----
    // Rejects unknown slots, overridden slots, and unlisted options; float
    // slots parse the value and clamp to their range.
    Result<void> set(const std::string& name, const std::string& value);
    Result<void> setFloat(const std::string& name, float value);

    // Slots whose public state changed since the last take (set/apply,
    // options, override flips) — the caller broadcasts these as events.
    std::vector<std::string> takeDirty();

private:
    struct Slot {
        std::string name;
        std::vector<std::string> options; // empty for float slots
        std::string active;               // choice slots
        std::string overriddenBy;
        bool isFloat = false;
        float value = 0.0f;
        float minValue = 0.0f;
        float maxValue = 0.0f;
        std::function<bool(const std::string&)> applyChoice;
        std::function<bool(float)> applyFloat;
        bool dirty = false;
    };

    Slot* find(const std::string& name);
    const Slot* find(const std::string& name) const;

    std::vector<Slot> slots_;
};

} // namespace rend::renderer

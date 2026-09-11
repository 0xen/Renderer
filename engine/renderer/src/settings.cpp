#include "rend/renderer/settings.h"

#include <algorithm>
#include <charconv>
#include <format>

namespace rend::renderer {

namespace {

bool listed(const std::vector<std::string>& options, const std::string& value) {
    return std::find(options.begin(), options.end(), value) != options.end();
}

std::string joined(const std::vector<std::string>& options) {
    std::string out;
    for (const std::string& option : options) {
        if (!out.empty()) {
            out += ", ";
        }
        out += option;
    }
    return out;
}

} // namespace

void SettingsRegistry::addChoice(std::string name, std::vector<std::string> options,
                                 std::string active,
                                 std::function<bool(const std::string&)> apply) {
    Slot slot;
    slot.name = std::move(name);
    slot.options = std::move(options);
    slot.active = std::move(active);
    slot.applyChoice = std::move(apply);
    slot.dirty = true; // initial state is broadcast-worthy
    slots_.push_back(std::move(slot));
}

void SettingsRegistry::addFloat(std::string name, float minValue, float maxValue, float value,
                                std::function<bool(float)> apply) {
    Slot slot;
    slot.name = std::move(name);
    slot.isFloat = true;
    slot.minValue = minValue;
    slot.maxValue = maxValue;
    slot.value = std::clamp(value, minValue, maxValue);
    slot.applyFloat = std::move(apply);
    slot.dirty = true;
    slots_.push_back(std::move(slot));
}

void SettingsRegistry::setOptions(const std::string& name, std::vector<std::string> options) {
    if (Slot* slot = find(name); slot && !slot->isFloat) {
        slot->options = std::move(options);
        if (!slot->options.empty() && !listed(slot->options, slot->active)) {
            slot->active = slot->options.front();
        }
        slot->dirty = true;
    }
}

void SettingsRegistry::setOverride(const std::string& name, const std::string& source) {
    if (Slot* slot = find(name); slot && slot->overriddenBy != source) {
        slot->overriddenBy = source;
        slot->dirty = true;
    }
}

void SettingsRegistry::clearOverride(const std::string& name) {
    if (Slot* slot = find(name); slot && !slot->overriddenBy.empty()) {
        slot->overriddenBy.clear();
        slot->dirty = true;
    }
}

std::vector<std::string> SettingsRegistry::names() const {
    std::vector<std::string> out;
    out.reserve(slots_.size());
    for (const Slot& slot : slots_) {
        out.push_back(slot.name);
    }
    return out;
}

bool SettingsRegistry::exists(const std::string& name) const { return find(name) != nullptr; }

std::vector<std::string> SettingsRegistry::options(const std::string& name) const {
    const Slot* slot = find(name);
    if (!slot || !slot->overriddenBy.empty()) {
        return {};
    }
    return slot->options;
}

std::string SettingsRegistry::active(const std::string& name) const {
    const Slot* slot = find(name);
    if (!slot) {
        return {};
    }
    if (!slot->overriddenBy.empty()) {
        return kSettingOverride;
    }
    if (slot->isFloat) {
        return std::format("{:.3f}", slot->value);
    }
    return slot->active;
}

std::string SettingsRegistry::overrideSource(const std::string& name) const {
    const Slot* slot = find(name);
    return slot ? slot->overriddenBy : std::string{};
}

bool SettingsRegistry::isFloat(const std::string& name) const {
    const Slot* slot = find(name);
    return slot && slot->isFloat;
}

float SettingsRegistry::floatValue(const std::string& name) const {
    const Slot* slot = find(name);
    return slot && slot->isFloat ? slot->value : 0.0f;
}

float SettingsRegistry::floatMin(const std::string& name) const {
    const Slot* slot = find(name);
    return slot && slot->isFloat ? slot->minValue : 0.0f;
}

float SettingsRegistry::floatMax(const std::string& name) const {
    const Slot* slot = find(name);
    return slot && slot->isFloat ? slot->maxValue : 0.0f;
}

Result<void> SettingsRegistry::set(const std::string& name, const std::string& value) {
    Slot* slot = find(name);
    if (!slot) {
        return Error{std::format("unknown setting '{}'", name)};
    }
    if (!slot->overriddenBy.empty()) {
        return Error{std::format("'{}' is overridden by '{}'", name, slot->overriddenBy)};
    }
    if (slot->isFloat) {
        float parsed = 0.0f;
        const auto [end, ec] =
            std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (ec != std::errc{} || end != value.data() + value.size()) {
            return Error{std::format("'{}' expects a number, got '{}'", name, value)};
        }
        return setFloat(name, parsed);
    }
    if (!listed(slot->options, value)) {
        return Error{std::format("'{}' has no option '{}' (available: {})", name, value,
                                 joined(slot->options))};
    }
    if (slot->active == value) {
        return {}; // no-op, no apply, no broadcast
    }
    if (slot->applyChoice && !slot->applyChoice(value)) {
        return Error{std::format("'{}' rejected '{}'", name, value)};
    }
    slot->active = value;
    slot->dirty = true;
    return {};
}

Result<void> SettingsRegistry::setFloat(const std::string& name, float value) {
    Slot* slot = find(name);
    if (!slot) {
        return Error{std::format("unknown setting '{}'", name)};
    }
    if (!slot->overriddenBy.empty()) {
        return Error{std::format("'{}' is overridden by '{}'", name, slot->overriddenBy)};
    }
    if (!slot->isFloat) {
        return Error{std::format("'{}' is not a numeric setting", name)};
    }
    const float clamped = std::clamp(value, slot->minValue, slot->maxValue);
    if (clamped == slot->value) {
        return {};
    }
    if (slot->applyFloat && !slot->applyFloat(clamped)) {
        return Error{std::format("'{}' rejected {}", name, value)};
    }
    slot->value = clamped;
    slot->dirty = true;
    return {};
}

std::vector<std::string> SettingsRegistry::takeDirty() {
    std::vector<std::string> out;
    for (Slot& slot : slots_) {
        if (slot.dirty) {
            out.push_back(slot.name);
            slot.dirty = false;
        }
    }
    return out;
}

SettingsRegistry::Slot* SettingsRegistry::find(const std::string& name) {
    for (Slot& slot : slots_) {
        if (slot.name == name) {
            return &slot;
        }
    }
    return nullptr;
}

const SettingsRegistry::Slot* SettingsRegistry::find(const std::string& name) const {
    for (const Slot& slot : slots_) {
        if (slot.name == name) {
            return &slot;
        }
    }
    return nullptr;
}

} // namespace rend::renderer

#pragma once

#include <filesystem>
#include <format>
#include <string_view>

namespace rend::log {

enum class Level { Trace, Info, Warn, Error };

// Backend-agnostic sink; currently stdout with timestamps and level tags.
void message(Level level, std::string_view text);

// Mirror every log line to a file as well (append; each line flushed so the
// log survives a crash or hang). Returns false if the file cannot be opened.
bool mirrorToFile(const std::filesystem::path& path);

template <typename... Args>
void trace(std::format_string<Args...> fmt, Args&&... args) {
    message(Level::Trace, std::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
void info(std::format_string<Args...> fmt, Args&&... args) {
    message(Level::Info, std::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
void warn(std::format_string<Args...> fmt, Args&&... args) {
    message(Level::Warn, std::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
void error(std::format_string<Args...> fmt, Args&&... args) {
    message(Level::Error, std::format(fmt, std::forward<Args>(args)...));
}

} // namespace rend::log

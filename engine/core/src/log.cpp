#include "rend/core/log.h"

#include <chrono>
#include <cstdio>

namespace rend::log {

namespace {

std::FILE* mirrorFile = nullptr;

const char* levelTag(Level level) {
    switch (level) {
    case Level::Trace: return "TRACE";
    case Level::Info: return "INFO ";
    case Level::Warn: return "WARN ";
    case Level::Error: return "ERROR";
    }
    return "?????";
}
} // namespace

void message(Level level, std::string_view text) {
    using namespace std::chrono;
    const auto now = time_point_cast<milliseconds>(system_clock::now());
    auto line = std::format("[{:%H:%M:%S}] [{}] {}\n", now, levelTag(level), text);
    std::FILE* stream = level == Level::Error ? stderr : stdout;
    std::fputs(line.c_str(), stream);
    // Redirected stdout is block-buffered; flush so a log survives a hang.
    std::fflush(stream);
    if (mirrorFile) {
        std::fputs(line.c_str(), mirrorFile);
        std::fflush(mirrorFile);
    }
}

bool mirrorToFile(const std::filesystem::path& path) {
    if (mirrorFile) {
        std::fclose(mirrorFile);
        mirrorFile = nullptr;
    }
#ifdef _WIN32
    mirrorFile = _wfopen(path.c_str(), L"a");
#else
    mirrorFile = std::fopen(path.c_str(), "a");
#endif
    if (!mirrorFile) {
        warn("Could not open log file '{}'", path.string());
        return false;
    }
    info("Log mirrored to '{}'", path.string());
    return true;
}

} // namespace rend::log

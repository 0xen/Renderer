#include "rend/core/log.h"

#include <chrono>
#include <cstdio>

namespace rend::log {

namespace {
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
}

} // namespace rend::log

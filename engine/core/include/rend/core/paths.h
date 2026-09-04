#pragma once

#include <filesystem>

namespace rend {

// Directory containing the running executable. Data artifacts (compiled
// shaders, scenes) are resolved relative to it, never to the working
// directory, so launching from anywhere behaves the same.
std::filesystem::path executableDirectory();

} // namespace rend

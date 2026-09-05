#pragma once

// Thin wrapper over the Tracy profiler client. Instrumented code writes
// these macros everywhere; they compile to nothing — and pull in no Tracy
// headers — unless the build sets REND_TRACY (opt-in CMake option).
#ifdef REND_TRACY
#include <tracy/Tracy.hpp>

// Named CPU zone scoped to the enclosing block; name must be a literal.
#define REND_PROFILE_ZONE(name) ZoneScopedN(name)
// Frame boundary; call once per frame, after present.
#define REND_PROFILE_FRAME() FrameMark
#else
#define REND_PROFILE_ZONE(name)
#define REND_PROFILE_FRAME()
#endif

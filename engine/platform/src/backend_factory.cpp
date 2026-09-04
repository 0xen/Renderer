#include "rend/platform/backend.h"

namespace rend::platform {

std::unique_ptr<IPlatformBackend> makeSdl3Backend(); // backends/sdl3

Result<std::unique_ptr<IPlatformBackend>> createBackend(BackendKind kind) {
    switch (kind) {
    case BackendKind::SDL3:
        return makeSdl3Backend();
    }
    return Error{"Unknown platform backend"};
}

} // namespace rend::platform

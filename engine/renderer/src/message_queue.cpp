#include "rend/renderer/message_queue.h"

#include <utility>

namespace rend::renderer {

void Sender::flush() {
    if (!staged_.empty()) {
        queue_->commit(staged_);
    }
}

void MessageQueue::commit(std::vector<Command>& batch) {
    std::lock_guard lock(commandMutex_);
    committed_.insert(committed_.end(), batch.begin(), batch.end());
    batch.clear();
}

std::vector<Command> MessageQueue::drain() {
    std::lock_guard lock(commandMutex_);
    return std::exchange(committed_, {});
}

void MessageQueue::pushEvent(const Event& event) {
    std::lock_guard lock(eventMutex_);
    events_.push_back(event);
}

std::vector<Event> MessageQueue::pollEvents() {
    std::lock_guard lock(eventMutex_);
    return std::exchange(events_, {});
}

} // namespace rend::renderer

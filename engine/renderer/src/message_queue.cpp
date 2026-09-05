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

std::vector<Event> EventReceiver::poll() {
    std::lock_guard lock(state_->mutex);
    return std::exchange(state_->events, {});
}

EventReceiver MessageQueue::createEventReceiver() {
    auto state = std::make_shared<EventReceiver::State>();
    std::lock_guard lock(receiverMutex_);
    receivers_.push_back(state);
    return EventReceiver(std::move(state));
}

void MessageQueue::pushEvent(const Event& event) {
    std::lock_guard lock(receiverMutex_);
    std::erase_if(receivers_, [](const auto& weak) { return weak.expired(); });
    for (const auto& weak : receivers_) {
        if (const auto state = weak.lock()) {
            std::lock_guard stateLock(state->mutex);
            state->events.push_back(event);
        }
    }
}

} // namespace rend::renderer

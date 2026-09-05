#pragma once

#include "rend/renderer/messages.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace rend::renderer {

class MessageQueue;

// One producer's staging area. push() buffers commands invisibly; flush()
// commits the whole batch atomically — the consumer sees all of it at one
// drain or none of it, so a load + place pair can never apply half.
// A Sender is NOT thread-safe; give each producer thread its own (the
// queue side is safe to share).
class Sender {
public:
    void push(const Command& command) { staged_.push_back(command); }
    void flush();

private:
    friend class MessageQueue;
    explicit Sender(MessageQueue& queue) : queue_(&queue) {}

    MessageQueue* queue_ = nullptr;
    std::vector<Command> staged_;
};

// One consumer's view of the event stream. Every pushEvent is broadcast
// to every live receiver (each producer — the app UI, the Python host —
// wants to see completions independently; a single shared poll would let
// consumers steal each other's events). poll() is thread-safe against
// pushEvent; a receiver itself belongs to one thread. Create receivers
// BEFORE issuing commands — events push only to receivers alive then.
class EventReceiver {
public:
    std::vector<Event> poll();

private:
    friend class MessageQueue;
    struct State {
        std::mutex mutex;
        std::vector<Event> events;
    };
    explicit EventReceiver(std::shared_ptr<State> state) : state_(std::move(state)) {}

    std::shared_ptr<State> state_;
};

// Command/event pair between producers (app, Python host) and the frame
// loop. Deliberately mutex-based, not lock-free: the consumer drains once
// per frame at its safe point (after the frame-slot fence), so contention
// is one brief lock per flush/drain — never a stall for either side.
class MessageQueue {
public:
    Sender createSender() { return Sender(*this); }
    EventReceiver createEventReceiver();

    // Producer-side unique model identities (0 is reserved as invalid).
    ModelHandle allocateHandle() { return nextHandle_.fetch_add(1); }

    // Consumer: takes every committed batch, in commit order.
    std::vector<Command> drain();

    // Consumer replies; producers poll their receivers. Same visibility
    // contract in reverse (events are singles, not batches — each is
    // complete). Destroyed receivers are pruned lazily here.
    void pushEvent(const Event& event);

private:
    friend class Sender;
    void commit(std::vector<Command>& batch);

    std::mutex commandMutex_;
    std::vector<Command> committed_;
    std::mutex receiverMutex_;
    std::vector<std::weak_ptr<EventReceiver::State>> receivers_;
    std::atomic<ModelHandle> nextHandle_{1};
};

} // namespace rend::renderer

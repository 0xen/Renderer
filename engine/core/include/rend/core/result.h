#pragma once

#include <string>
#include <utility>
#include <variant>

namespace rend {

// Error value carried across module boundaries instead of exceptions.
struct Error {
    std::string message;
};

// Minimal expected-style result. Holds either a T or an Error.
template <typename T>
class Result {
public:
    Result(T value) : storage_(std::move(value)) {}
    Result(Error error) : storage_(std::move(error)) {}

    bool ok() const { return std::holds_alternative<T>(storage_); }
    explicit operator bool() const { return ok(); }

    T& value() & { return std::get<T>(storage_); }
    const T& value() const& { return std::get<T>(storage_); }
    T&& value() && { return std::get<T>(std::move(storage_)); }

    const Error& error() const { return std::get<Error>(storage_); }

private:
    std::variant<T, Error> storage_;
};

template <>
class Result<void> {
public:
    Result() = default;
    Result(Error error) : error_(std::move(error)), failed_(true) {}

    bool ok() const { return !failed_; }
    explicit operator bool() const { return ok(); }

    const Error& error() const { return error_; }

private:
    Error error_;
    bool failed_ = false;
};

} // namespace rend

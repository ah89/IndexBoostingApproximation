#include "buffer_manager.h"

BufferManager::BufferManager(size_t buffer_size, size_t batch_size)
    : buffer_size_(buffer_size), batch_size_(batch_size) {}

void BufferManager::add(double key, size_t value) {
    buffer_.emplace_back(key, value);
    if (buffer_.size() > buffer_size_) {
        buffer_.erase(buffer_.begin());
    }
}

bool BufferManager::is_full() const {
    return buffer_.size() >= batch_size_;
}

std::pair<std::vector<double>, std::vector<size_t>> BufferManager::get_batch() const {
    std::vector<double> keys;
    std::vector<size_t> values;
    keys.reserve(std::min(batch_size_, buffer_.size()));
    values.reserve(std::min(batch_size_, buffer_.size()));

    for (size_t i = 0; i < std::min(batch_size_, buffer_.size()); ++i) {
        keys.push_back(buffer_[i].first);
        values.push_back(static_cast<size_t>(buffer_[i].second));
    }

    return {keys, values};
}

void BufferManager::clear() {
    buffer_.clear();
}
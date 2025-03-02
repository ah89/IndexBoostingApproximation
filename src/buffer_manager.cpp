#include "buffer_manager.h"

BufferManager::BufferManager(size_t buffer_size, size_t batch_size)
    : buffer_size_(buffer_size), batch_size_(batch_size) {}

void BufferManager::add(double key, size_t value) {
    buffer_[key] = value;

    // Ensure we do not exceed buffer size
    if (buffer_.size() > buffer_size_) {
        buffer_.erase(buffer_.begin());  // Remove the smallest (oldest) key
    }
}

bool BufferManager::is_full() const {
    return buffer_.size() >= batch_size_;
}

bool BufferManager::possible_to_add(size_t new_data_size) const {
    return buffer_.size() + new_data_size >= batch_size_;
}

size_t BufferManager::get_size() const {
    return buffer_.size();
}

std::pair<std::vector<double>, std::vector<size_t>> BufferManager::get_batch() const {
    std::vector<double> keys;
    std::vector<size_t> values;
    
    keys.reserve(std::min(batch_size_, buffer_.size()));
    values.reserve(std::min(batch_size_, buffer_.size()));

    size_t count = 0;
    for (const auto& [key, value] : buffer_) {
        if (count >= batch_size_) break;
        keys.push_back(key);
        values.push_back(value);
        count++;
    }

    return {keys, values};
}

void BufferManager::clear() {
    buffer_.clear();
}
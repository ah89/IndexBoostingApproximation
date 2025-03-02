#pragma once

#include <map>
#include <vector>
#include <algorithm>

class BufferManager {
public:
    BufferManager(size_t buffer_size, size_t batch_size);
    
    void add(double key, size_t value);
    bool is_full() const;
    bool possible_to_add(size_t new_data_size) const;
    size_t get_size() const;
    
    std::pair<std::vector<double>, std::vector<size_t>> get_batch() const;
    void clear();

private:
    size_t buffer_size_;
    size_t batch_size_;
    std::map<double, size_t> buffer_;  // B+Tree-like sorted structure
};
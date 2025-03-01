#pragma once

#include <vector>
#include <algorithm>
#include "radix_spline.h"

class BufferManager {
public:
    BufferManager(size_t buffer_size, size_t batch_size);
    void add(double key, size_t value);
    bool possible_to_add(size_t new_data_size) const;
    bool is_full() const;
    std::pair<std::vector<double>, std::vector<size_t>> get_batch() const;
    void clear();
    size_t get_size() const;

private:
    size_t buffer_size_;
    size_t batch_size_;
    std::vector<std::pair<double, size_t>> buffer_;
};


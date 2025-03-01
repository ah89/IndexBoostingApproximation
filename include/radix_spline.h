// include/radix_spline.h
#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

class RadixSpline {
public:
    RadixSpline(int num_radix_bits, int max_error);
    void build(const std::vector<double>& keys, const std::vector<size_t>& positions);
    size_t predict(double key) const;

private:
    struct Spline {
        double key;
        double slope;
        double intercept;
    };

    int num_radix_bits_;
    int max_error_;
    std::vector<Spline> splines_;
    std::vector<size_t> radix_table_;

    size_t get_radix_index(double key) const;
    void update_num_radix_bits(int num_radix_bits){
        num_radix_bits_ = ceil(log2(num_radix_bits));
        radix_table_.resize(1 << num_radix_bits_);
    }
};
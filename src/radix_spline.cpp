// src/radix_spline.cpp
#include "../include/radix_spline.h"

RadixSpline::RadixSpline(int num_radix_bits, int max_error)
    : num_radix_bits_(num_radix_bits), max_error_(max_error)
{
    radix_table_.resize(1 << num_radix_bits_);
}

void RadixSpline::build(const std::vector<double> &keys, const std::vector<size_t> &positions)
{

    update_num_radix_bits(keys.size());
    // Ensure inputs are valid
    size_t n = keys.size();
    if (n < 2)
    {
        throw std::invalid_argument("Number of keys must be at least 2.");
    }

    splines_.clear();
    splines_.push_back({keys[0], 0, static_cast<double>(positions[0])});

    // Start of the current spline segment
    size_t segment_start_index = 0;
    double segment_start_key = keys[0];
    double segment_start_position = positions[0];
    for (size_t i = 1; i < n; ++i)
    {
        if (keys[i] != -1)
        {
            segment_start_key = keys[i];
            segment_start_position = positions[i];
            segment_start_index = i;
            break;
        }
    }

    for (size_t i = segment_start_index; i < n; ++i)
    {
        // Continue on the placeholders
        if (keys[i] == -1)
            continue;

        // Calculate the slope of the line from the start of the segment to the current point

        double current_slope = (positions[i] - segment_start_position) / (keys[i] - segment_start_key);

        // Check the maximum interpolation error for all points in the current segment
        bool within_error_bound = true;
        for (size_t j = splines_.back().key == segment_start_key ? static_cast<size_t>(splines_.back().intercept) : 0; j <= i; ++j)
        {
            double predicted_position = current_slope * (keys[j] - segment_start_key) + segment_start_position;
            double actual_position = static_cast<double>(positions[j]);
            double error = std::abs(predicted_position - actual_position);

            if (error > max_error_)
            {
                within_error_bound = false;
                break;
            }
        }

        if (!within_error_bound)
        {
            splines_.push_back({keys[i - 1], current_slope, static_cast<double>(positions[i - 1])});
            segment_start_key = keys[i - 1];
            segment_start_position = positions[i - 1];
        }
    }

    // Add the last spline segment
    double final_slope = (positions.back() - segment_start_position) / (keys.back() - segment_start_key);
    splines_.push_back({keys.back(), final_slope, static_cast<double>(positions.back())});

    // Build the radix table
    for (size_t i = 0; i < radix_table_.size(); ++i)
    {
        double radix_key = static_cast<double>(i) / radix_table_.size() * (keys.back() - keys.front()) + keys.front();
        radix_table_[i] = std::lower_bound(splines_.begin(), splines_.end(), radix_key,
                                           [](const Spline &s, double k)
                                           { return s.key < k; }) -
                          splines_.begin();
    }
    std::cout << "The tadix table built" << std::endl;
}

size_t RadixSpline::predict(double key) const
{
    size_t radix_index = get_radix_index(key);
    size_t spline_index = radix_table_[radix_index];
    const Spline &spline = splines_[spline_index];
    return static_cast<size_t>(spline.slope * (key - spline.key) + spline.intercept);
}

size_t RadixSpline::get_radix_index(double key) const
{
    return static_cast<size_t>((key - splines_.front().key) / (splines_.back().key - splines_.front().key) * radix_table_.size());
}
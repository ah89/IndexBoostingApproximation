#include "placeholder_strategy.h"
#include <algorithm>
#include <random>

PlaceholderStrategy::PlaceholderStrategy(int num_placeholders)
    : num_placeholders_(num_placeholders) {}

std::vector<double> PlaceholderStrategy::insert_placeholders(const std::vector<double>& keys, const GMM& gmm) {
    std::vector<double> result = keys;
    std::vector<double> placeholders;

    // Generate placeholders based on GMM prediction
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    for (int i = 0; i < num_placeholders_; ++i) {
        double x = dis(gen);
        double y = gmm.predict(x);
        placeholders.push_back(y!=0? y : x * (keys.back() - keys[0]));
    }

    // Sort placeholders
    std::sort(placeholders.begin(), placeholders.end());

    // Merge placeholders with current keys
    std::vector<double> merged;
    merged.reserve(result.size() + placeholders.size());

    auto it_keys = result.begin();
    auto it_placeholders = placeholders.begin();

    size_t cntr(0);

    while (cntr < result.size() + placeholders.size()) {
        if (*it_keys < *it_placeholders) {
            merged.push_back(*it_keys++);
        } else {
            merged.push_back(-1);
            ++it_placeholders;
        }
        cntr ++;
    }

    // merged.insert(merged.end(), it_keys, result.end());
    // merged.insert(merged.end(), it_placeholders, placeholders.end());
    return merged;
}
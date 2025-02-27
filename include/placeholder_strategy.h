#pragma once

#include <vector>
#include "gmm.h"

class PlaceholderStrategy {
public:
    PlaceholderStrategy(int num_placeholders);
    std::vector<double> insert_placeholders(const std::vector<double>& keys, const GMM& gmm);

private:
    int num_placeholders_;
};
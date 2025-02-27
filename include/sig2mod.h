#pragma once

#include <vector>
#include <memory>
#include "radix_spline.h"
#include "neural_network.h"
#include "gmm.h"
#include "buffer_manager.h"
#include "control_unit.h"
#include "placeholder_strategy.h"

class Sig2Mod {
public:
    Sig2Mod(std::unique_ptr<ComplexNN> complex_nn,
            double error_range,
            size_t buffer_size,
            size_t batch_size,
            double error_threshold,
            int max_iterations,
            int num_placeholders);

    void insert(const std::vector<double>& keys, const std::vector<double>& values);
    double lookup(double key);
    void update(double key, double value);
    void train();

private:
    std::unique_ptr<RadixSpline> learned_index_;
    std::unique_ptr<ComplexNN> complex_nn_;
    std::unique_ptr<GMM> gmm_;
    std::unique_ptr<BufferManager> buffer_manager_;
    std::unique_ptr<ControlUnit> control_unit_;
    std::unique_ptr<PlaceholderStrategy> placeholder_strategy_;
    double error_range_;
};
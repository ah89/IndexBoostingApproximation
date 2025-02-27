#pragma once

#include "radix_spline.h"
#include "neural_network.h"
#include "gmm.h"
#include "buffer_manager.h"

class ControlUnit {
public:
    ControlUnit(double error_threshold, int max_iterations);
    bool should_retrain(const RadixSpline& learned_index, const ComplexNN& complex_nn, const GMM& gmm, const BufferManager& buffer);
    void trigger_retraining(RadixSpline& learned_index, ComplexNN& complex_nn, GMM& gmm, BufferManager& buffer);

private:
    double error_threshold_;
    int max_iterations_;
    int current_iterations_;
};
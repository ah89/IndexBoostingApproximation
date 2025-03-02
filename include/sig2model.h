#pragma once

#include <vector>
#include <memory>
#include <optional>
#include "radix_spline.h"
#include "neural_network.h"
#include "gmm.h"
#include "buffer_manager.h"
#include "control_unit.h"
#include "placeholder_strategy.h"

class SigmaSigmoid {
    public:
        SigmaSigmoid(size_t max_sigmoids);
        void update(const std::vector<double>& new_sigmoids);
        double adjust(double key) const;
        bool hasUpdates() const;
    
    private:
        struct SigmoidParams {
            double A;      // Amplitude
            double omega;  // Slope
            double phi;    // Center
        };
    
        std::vector<SigmoidParams> sigmoids_;
        bool has_updates_;
    };


class Sig2Model {
    public:
        Sig2Model(
            double error_range,
            size_t buffer_size,
            size_t batch_size,
            double error_threshold,
            int max_iterations,
            int K,
            int N,
            int num_placeholders);

        void insert(const std::vector<double> &keys, const std::vector<double> &values);
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
        std::unique_ptr<SigmaSigmoid> sigma_sigmoid_;

        double error_range_;

        // Keep ordered keys and corresponding values
        std::vector<double> keys_;
        std::vector<double> values_;
};
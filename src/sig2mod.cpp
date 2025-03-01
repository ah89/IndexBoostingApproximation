#include "sig2mod.h"
#include <algorithm>
#include <cmath>

Sig2Mod::Sig2Mod(std::unique_ptr<ComplexNN> complex_nn,
                 double error_range,
                 size_t buffer_size,
                 size_t batch_size,
                 double error_threshold,
                 int max_iterations,
                 int num_placeholders)
    : complex_nn_(std::move(complex_nn)),
      error_range_(error_range) {
    learned_index_ = std::make_unique<RadixSpline>(10, static_cast<int>(error_range));
    gmm_ = std::make_unique<GMM>(5);
    buffer_manager_ = std::make_unique<BufferManager>(buffer_size, batch_size);
    control_unit_ = std::make_unique<ControlUnit>(error_threshold, max_iterations);
    placeholder_strategy_ = std::make_unique<PlaceholderStrategy>(num_placeholders);
}

void Sig2Mod::insert(const std::vector<double>& keys, const std::vector<double>& values) {
    // std::vector<double> keys_with_placeholders = placeholder_strategy_->insert_placeholders(keys, *gmm_);
    std::vector<double> keys_with_placeholders = keys;

    std::vector<size_t> positions(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        positions[i] = i;
    }
    
    learned_index_->build(keys_with_placeholders, positions);
    
    gmm_->fit(keys);
    
    // Train the ComplexNN
    std::vector<std::vector<double>> X_pi(keys.size(), std::vector<double>(1));
    std::vector<std::vector<double>> X_phi(keys.size(), std::vector<double>(1));
    std::vector<std::vector<double>> y_pi(keys.size(), std::vector<double>(3));
    std::vector<std::vector<double>> y_phi(keys.size(), std::vector<double>(3));
    
    for (size_t i = 0; i < keys.size(); ++i) {
        X_pi[i][0] = keys[i];
        X_phi[i][0] = keys[i];
        // Initialize y_pi and y_phi with zeros (simplified)
        std::fill(y_pi[i].begin(), y_pi[i].end(), 0.0);
        std::fill(y_phi[i].begin(), y_phi[i].end(), 0.0);
    }
    
    complex_nn_->train(X_pi, X_phi, positions, 100, 0.01);
}

double Sig2Mod::lookup(double key) {
    size_t predicted_index = learned_index_->predict(key);
    std::vector<double> nn_input_pi = {key};
    std::vector<double> nn_input_phi = {key};
    auto [pi_output, phi_output] = complex_nn_->forward(nn_input_pi, nn_input_phi);
    
    // Apply sigmoid adjustments (simplified)
    for (size_t i = 0; i < pi_output.size() / 3; ++i) {
        double A = pi_output[i * 3];
        double omega = pi_output[i * 3 + 1];
        double phi = pi_output[i * 3 + 2];
        
        predicted_index += static_cast<size_t>(A / (1 + std::exp(-omega * (key - phi))));
    }
    
    return static_cast<double>(predicted_index);
}

void Sig2Mod::update(double key, double value) {
    buffer_manager_->add(key, value);
    if (buffer_manager_->is_full()) {
        if (control_unit_->should_retrain(*learned_index_, *complex_nn_, *gmm_, *buffer_manager_)) {
            train();
        }
    }
}

void Sig2Mod::train() {
    control_unit_->trigger_retraining(*learned_index_, *complex_nn_, *gmm_, *buffer_manager_);
    
    auto batch = buffer_manager_->get_batch();
    std::vector<double> keys_with_placeholders = placeholder_strategy_->insert_placeholders(batch.first, *gmm_);
    
    std::vector<size_t> positions(keys_with_placeholders.size());
    for (size_t i = 0; i < keys_with_placeholders.size(); ++i) {
        positions[i] = i;
    }
    learned_index_->build(keys_with_placeholders, positions);
}
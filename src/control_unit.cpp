#include "control_unit.h"

ControlUnit::ControlUnit(double error_threshold, int max_iterations)
    : error_threshold_(error_threshold), max_iterations_(max_iterations), current_iterations_(0) {}

bool ControlUnit::should_retrain(const RadixSpline& learned_index, const ComplexNN& complex_nn, const GMM& gmm, const BufferManager& buffer) {
    if (buffer.is_full()) {
        ++current_iterations_;
    }

    if (current_iterations_ >= max_iterations_) {
        return true;
    }

    auto batch = buffer.get_batch();
    double total_error = 0;
    for (size_t i = 0; i < batch.first.size(); ++i) {
        double predicted = learned_index.predict(batch.first[i]);
        double actual = static_cast<double>(batch.second[i]);
        total_error += std::abs(predicted - actual);
    }
    double avg_error = total_error / batch.first.size();

    return avg_error > error_threshold_;
}

void ControlUnit::trigger_retraining(RadixSpline& learned_index, ComplexNN& complex_nn, GMM& gmm, BufferManager& buffer) {
    auto batch = buffer.get_batch();

    learned_index.build(batch.first, batch.second);

    std::vector<std::vector<double>> X_pi(batch.first.size(), std::vector<double>(1));
    std::vector<std::vector<double>> X_phi(batch.first.size(), std::vector<double>(1));
    std::vector<std::vector<double>> y_pi(batch.first.size(), std::vector<double>(3));
    std::vector<std::vector<double>> y_phi(batch.first.size(), std::vector<double>(3));

    for (size_t i = 0; i < batch.first.size(); ++i) {
        X_pi[i][0] = batch.first[i];
        X_phi[i][0] = batch.first[i];
        
        // Compute targets for y_pi and y_phi (simplified)
        y_pi[i] = {0, 0, 0};
        y_phi[i] = {0, 0, 0};
    }

    complex_nn.train(X_pi, X_phi, y_pi, y_phi, 100, 0.01);

    gmm.fit(batch.first);

    buffer.clear();
    current_iterations_ = 0;
}
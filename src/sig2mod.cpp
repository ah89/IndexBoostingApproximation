#include "sig2mod.h"
#include <algorithm>
#include <cmath>

// SigmaSigmoid Implementation
SigmaSigmoid::SigmaSigmoid(size_t max_sigmoids) : has_updates_(false)
{
    sigmoids_.resize(max_sigmoids, {0.0, 0.0, 0.0});
}

void SigmaSigmoid::update(const std::vector<double> &new_sigmoids)
{
    if (new_sigmoids.size() % 3 != 0)
    {
        throw std::invalid_argument("Sigmoids must be provided in triples (A, omega, phi).");
    }
    size_t num_sigmoids = new_sigmoids.size() / 3;
    for (size_t i = 0; i < num_sigmoids; ++i)
    {
        sigmoids_[i] = {new_sigmoids[i * 3], new_sigmoids[i * 3 + 1], new_sigmoids[i * 3 + 2]};
    }
    has_updates_ = true;
}

double SigmaSigmoid::adjust(double key) const
{
    if (!has_updates_)
        return 0.0; // No updates, return zero adjustment

    double adjustment = 0.0;
    for (const auto &sigmoid : sigmoids_)
    {
        if (sigmoid.A != 0)
        {
            adjustment += sigmoid.A / (1 + std::exp(-sigmoid.omega * (key - sigmoid.phi)));
        }
    }
    return adjustment;
}

bool SigmaSigmoid::hasUpdates() const
{
    return has_updates_;
}

Sig2Mod::Sig2Mod(
    double error_range,
    size_t buffer_size,
    size_t batch_size,
    double error_threshold,
    int max_iterations,
    int K,
    int N,
    int num_placeholders)
    : error_range_(error_range)
{
    complex_nn_ = std::make_unique<ComplexNN>(N, K, std::min(4 * (N + K), 50), 3 * (N + K));
    learned_index_ = std::make_unique<RadixSpline>(10, static_cast<int>(error_range));
    gmm_ = std::make_unique<GMM>(K);
    buffer_manager_ = std::make_unique<BufferManager>(buffer_size, batch_size);
    control_unit_ = std::make_unique<ControlUnit>(error_threshold, max_iterations);
    placeholder_strategy_ = std::make_unique<PlaceholderStrategy>(num_placeholders);
    sigma_sigmoid_ = std::make_unique<SigmaSigmoid>(N);
}

void Sig2Mod::insert(const std::vector<double> &keys, const std::vector<double> &values)
{

    // For S2M-B, make this part comment
    // if(buffer_manager_->possible_to_add(keys.size())){
    //     for(size_t i =0; i<keys.size(); i++){
    //         buffer_manager_->add(keys[i],values[i]);
    //     }
    //     return;
    // }
    //

    // std::vector<double> keys_with_placeholders = placeholder_strategy_->insert_placeholders(keys, *gmm_);
    std::vector<double> keys_with_placeholders = keys;

    std::vector<size_t> positions(keys.size());
    for (size_t i = 0; i < keys.size(); ++i)
    {
        positions[i] = i;
    }

    learned_index_->build(keys_with_placeholders, positions);

    gmm_->fit(keys);

    // Train the ComplexNN
    std::vector<std::vector<double>> X_pi(keys.size(), std::vector<double>(1));
    std::vector<std::vector<double>> X_phi(keys.size(), std::vector<double>(1));
    std::vector<std::vector<double>> y_pi(keys.size(), std::vector<double>(3));
    std::vector<std::vector<double>> y_phi(keys.size(), std::vector<double>(3));

    for (size_t i = 0; i < keys.size(); ++i)
    {
        X_pi[i][0] = keys[i];
        X_phi[i][0] = keys[i];
    }

    complex_nn_->train(X_pi, X_phi, positions, 100, 0.01);
}

double Sig2Mod::lookup(double key)
{
    size_t predicted_index = learned_index_->predict(key);
    std::vector<double> nn_input_pi = {key};
    std::vector<double> nn_input_phi = {key};
    if (sigma_sigmoid_->hasUpdates())
        auto [pi_output, phi_output] = complex_nn_->forward(nn_input_pi, nn_input_phi);

    predicted_index += sigma_sigmoid_->adjust(key);

    return static_cast<double>(predicted_index);
}

void Sig2Mod::update(double key, double value)
{
    buffer_manager_->add(key, value);
    if (buffer_manager_->is_full())
    {
        if (control_unit_->should_retrain(*learned_index_, *complex_nn_, *gmm_, *buffer_manager_))
        {
            train();
        }
    }
}

void Sig2Mod::train()
{
    control_unit_->trigger_retraining(*learned_index_, *complex_nn_, *gmm_, *buffer_manager_);

    auto batch = buffer_manager_->get_batch();
    std::vector<double> keys_with_placeholders = placeholder_strategy_->insert_placeholders(batch.first, *gmm_);

    std::vector<size_t> positions(keys_with_placeholders.size());
    for (size_t i = 0; i < keys_with_placeholders.size(); ++i)
    {
        positions[i] = i;
    }
    learned_index_->build(keys_with_placeholders, positions);
}
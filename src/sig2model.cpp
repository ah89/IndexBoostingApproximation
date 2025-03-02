#include "sig2model.h"
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

Sig2Model::Sig2Model(
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

void Sig2Model::insert(const std::vector<double> &keys, const std::vector<double> &values)
{
    if (keys.empty() || values.empty() || keys.size() != values.size()) {
        throw std::invalid_argument("Keys and values must have the same non-zero size.");
    }

    // For S2M-B, make this part comment
    // if(buffer_manager_->possible_to_add(keys.size())){
    //     for(size_t i =0; i<keys.size(); i++){
    //         buffer_manager_->add(keys[i],values[i]);
    //     }
    //     return;
    // }
    //

    std::vector<double> merged_keys;
    std::vector<double> merged_values;

    // Merge existing keys and new keys while keeping order
    size_t i = 0, j = 0;
    while (i < keys_.size() && j < keys.size()) {
        if (keys_[i] < keys[j]) {
            merged_keys.push_back(keys_[i]);
            merged_values.push_back(values_[i]);
            i++;
        } else {
            merged_keys.push_back(keys[j]);
            merged_values.push_back(values[j]);
            j++;
        }
    }

    // Append remaining elements
    while (i < keys_.size()) {
        merged_keys.push_back(keys_[i]);
        merged_values.push_back(values_[i]);
        i++;
    }
    while (j < keys.size()) {
        merged_keys.push_back(keys[j]);
        merged_values.push_back(values[j]);
        j++;
    }

    // Update keys and values
    keys_ = std::move(merged_keys);
    values_ = std::move(merged_values);


    // For S2M-Ψ, make this part comment
    // std::vector<double> keys_with_placeholders = placeholder_strategy_->insert_placeholders(values, *gmm_);

    std::vector<double> keys_with_placeholders = keys_;

    std::vector<size_t> positions(keys_with_placeholders.size());
    for (size_t i = 0; i < keys_with_placeholders.size(); ++i)
    {
        positions[i] = i;
    }

    learned_index_->build(keys_with_placeholders, positions);

    gmm_->fit(keys);

    // Train the ComplexNN
    std::vector<std::vector<double>> X_pi(keys_with_placeholders.size(), std::vector<double>(1));
    std::vector<std::vector<double>> X_phi(keys_with_placeholders.size(), std::vector<double>(1));

    for (size_t i = 0; i < keys_with_placeholders.size(); ++i)
    {
        X_pi[i][0] = keys_with_placeholders[i];
        X_phi[i][0] = keys_with_placeholders[i];
    }

    complex_nn_->train(X_pi, X_phi, positions, 100, 0.01);
}

std::vector<double> Sig2Model::lookup(double key)
{
    std::vector<double> result_values;

    // Check the buffer first
    auto buffer_result = buffer_manager_->lookup(key);
    if (buffer_result.has_value()) {
        result_values.push_back(static_cast<double>(buffer_result.value()));
        return result_values;
    }

    // If not found in buffer, proceed to learned index lookup
    size_t predicted_index = learned_index_->predict(key);
    
    // Apply sigmoid adjustments
    if (sigma_sigmoid_->hasUpdates())
    {
        predicted_index += static_cast<size_t>(sigma_sigmoid_->adjust(key));
    }

    predicted_index = std::min(predicted_index, keys_.size() - 1);

    // Locate closest key range
    auto it = std::lower_bound(keys_.begin(), keys_.end(), key);
    size_t lower_bound = std::max(static_cast<int>(predicted_index) - static_cast<int>(error_range_), 0);
    size_t upper_bound = std::min(predicted_index + static_cast<size_t>(error_range_), values_.size() - 1);

    // Collect values in range
    for (size_t i = lower_bound; i <= upper_bound; ++i) {
        result_values.push_back(values_[i]);
    }

    return result_values;
}

void Sig2Model::update(double key, double value)
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

void Sig2Model::train()
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
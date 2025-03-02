#include <gtest/gtest.h>
#include "sig2model.h"
#include "radix_spline.h"
#include "neural_network.h"
#include "gmm.h"
#include "buffer_manager.h"
#include "control_unit.h"
#include "placeholder_strategy.h"
#include <random>
#include <algorithm>
#include <chrono>

class Sig2ModelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize Sig2Model with some default parameters
        int N = 3; // sigmoid_capacity
        int K = 3; // number of Gaussian components
        int hidden_dim = 15;
        int output_dim = 3 * (N + K);

        sig2model = std::make_unique<Sig2Model>(
            5.0,  // error_range
            5,    // buffer_size
            10,   // batch_size
            0.01, // error_threshold
            100,  // max_iterations
            K,
            N,
            15 // num_placeholders
        );
    }

    std::unique_ptr<Sig2Model> sig2model;

    // Helper function to generate random data
    std::pair<std::vector<double>, std::vector<double>> generate_random_data(size_t n, double min_key, double max_key)
    {
        std::vector<double> keys(n);
        std::vector<double> values(n);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> key_dis(min_key, max_key);
        std::uniform_real_distribution<> value_dis(0.0, 1000.0);

        for (size_t i = 0; i < n; ++i)
        {
            keys[i] = key_dis(gen);
            values[i] = value_dis(gen);
        }

        std::sort(keys.begin(), keys.end());
        return {keys, values};
    }
};

TEST_F(Sig2ModelTest, InsertAndLookup)
{
    auto [keys, values] = generate_random_data(1000, 0.0, 10000.0);

    // Insert data
    sig2model->insert(keys, values);

    // Test lookup
    for (size_t i = 0; i < keys.size(); ++i)
    {
        std::vector<double> results = sig2model->lookup(keys[i]);

        // Check if expected value is in results
        auto it = std::find(results.begin(), results.end(), values[i]);
        if (it == results.end())
        {
            throw std::runtime_error("Expected value not found in lookup results!");
        }
    }
}

TEST_F(Sig2ModelTest, Update) {
    auto [keys, values] = generate_random_data(10000, 0.0, 100000.0);

    // Insert initial data
    sig2model->insert(keys, values);

    // Perform updates and store expected values
    std::map<double, double> updated_values;
    for (size_t i = 0; i < 10; ++i) {
        double new_key = keys[i] + 0.5;
        double new_value = values[i] + 100.0;
        sig2model->update(new_key, new_value);
        updated_values[new_key] = new_value;
    }

    // Test lookup after updates
    for (const auto& [key, expected_value] : updated_values) {
        std::vector<double> results = sig2model->lookup(key);
        if (std::find(results.begin(), results.end(), expected_value) == results.end()) {
            throw std::runtime_error("Updated value not found in lookup results!");
        }
    }
}

TEST_F(Sig2ModelTest, TrainAndRetrain) {
    auto [keys, values] = generate_random_data(10000, 0.0, 100000.0);

    // Insert initial data
    sig2model->insert(keys, values);

    // Perform updates to trigger retraining
    for (size_t i = 0; i < 2000; ++i) {
        double new_key = 1001.0 + i;
        double new_value = 2000.0 + i;
        sig2model->update(new_key, new_value);
    }

    // Test lookup after retraining
    for (size_t i = 0; i < 1000; ++i) {
        double expected_value = 2000.0 + i;
        std::vector<double> results = sig2model->lookup(1001.0 + i);
        if (std::find(results.begin(), results.end(), expected_value) == results.end()) {
            throw std::runtime_error("Retrained value not found in lookup results!");
        }
    }
}

TEST_F(Sig2ModelTest, DistributionShift) {
    auto [keys1, values1] = generate_random_data(10000, 0.0, 100000.0);
    auto [keys2, values2] = generate_random_data(10000, 2000.0, 300000.0);

    // Insert initial data
    sig2model->insert(keys1, values1);

    // Insert new distribution data
    for (size_t i = 0; i < keys2.size(); ++i) {
        sig2model->update(keys2[i], values2[i]);
    }

    // Test lookup for both distributions
    for (size_t i = 0; i < 1000; ++i) {
        std::vector<double> results1 = sig2model->lookup(keys1[i]);
        std::vector<double> results2 = sig2model->lookup(keys2[i]);

        if (std::find(results1.begin(), results1.end(), values1[i]) == results1.end()) {
            throw std::runtime_error("Value from first distribution not found!");
        }

        if (std::find(results2.begin(), results2.end(), values2[i]) == results2.end()) {
            throw std::runtime_error("Value from second distribution not found!");
        }
    }
}

TEST_F(Sig2ModelTest, PerformanceBenchmark)
{
    auto [keys, values] = generate_random_data(1000000, 0.0, 100000000.0);

    // Measure insert time
    auto start = std::chrono::high_resolution_clock::now();
    sig2model->insert(keys, values);
    auto end = std::chrono::high_resolution_clock::now();
    auto insert_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Measure lookup time
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 10000; ++i)
    {
        size_t index = rand() % keys.size();
        std::vector<double> results = sig2model->lookup(keys[index]);  // Get list of values

        // Ensure at least one value is returned (avoid empty lookup results)
        ASSERT_FALSE(results.empty()) << "Lookup returned an empty result for key: " << keys[index];

        // Access the first value to simulate usage (force compiler to not optimize away)
        volatile double first_value = results[0];
    }
    end = std::chrono::high_resolution_clock::now();
    auto lookup_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Output performance metrics
    std::cout << "Insert time for 1M keys: " << insert_duration.count() << " ms" << std::endl;
    std::cout << "Average lookup time: " << lookup_duration.count() / 10000.0 << " µs" << std::endl;

    // Add performance expectations
    EXPECT_LT(insert_duration.count(), 10000);         // Expect insert to take < 10 seconds
    EXPECT_LT(lookup_duration.count() / 10000.0, 100); // Expect avg lookup < 100 µs
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
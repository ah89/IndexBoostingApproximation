#include <gtest/gtest.h>
#include "sig2mod.h"
#include "radix_spline.h"
#include "neural_network.h"
#include "gmm.h"
#include "buffer_manager.h"
#include "control_unit.h"
#include "placeholder_strategy.h"
#include <random>
#include <algorithm>
#include <chrono>

class Sig2ModTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize Sig2Mod with some default parameters
        int N = 3; // sigmoid_capacity
        int K = 3; // number of Gaussian components
        int hidden_dim = 15;
        int output_dim = 3 * (N + K);

        sig2mod = std::make_unique<Sig2Mod>(
            5.0,  // error_range
            5,    // buffer_size
            10,   // batch_size
            0.01, // error_threshold
            100,  // max_iterations
            K,
            N,
            15    // num_placeholders
        );
    }

    std::unique_ptr<Sig2Mod> sig2mod;

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

TEST_F(Sig2ModTest, InsertAndLookup)
{
    auto [keys, values] = generate_random_data(10, 0.0, 100.0);

    // Insert data
    sig2mod->insert(keys, values);

    // Test lookup
    for (size_t i = 0; i < keys.size(); ++i)
    {
        double result = sig2mod->lookup(keys[i]);
        std::cout << i << " " << result << std::endl;
        std::cout << result << std::endl;
        std::cout << "\n" << std::endl;
        EXPECT_NEAR(result, i, 5); // Allow for error within the specified range
    }
}

TEST_F(Sig2ModTest, Update)
{
    auto [keys, values] = generate_random_data(10000, 0.0, 100000.0);

    // Insert initial data
    sig2mod->insert(keys, values);

    // Perform updates
    for (size_t i = 0; i < 10; ++i)
    {
        double new_key = keys[i] + 0.5;
        double new_value = values[i] + 100.0;
        sig2mod->update(new_key, new_value);
    }

    // Test lookup after updates
    for (size_t i = 0; i < 10; ++i)
    {
        double result = sig2mod->lookup(keys[i] + 0.5);
        EXPECT_NEAR(result, i, 256.0); // Allow for larger error due to updates
    }
}

TEST_F(Sig2ModTest, TrainAndRetrain)
{
    auto [keys, values] = generate_random_data(10000, 0.0, 100000.0);

    // Insert initial data
    sig2mod->insert(keys, values);

    // Perform multiple updates to trigger retraining
    for (size_t i = 0; i < 2000; ++i)
    {
        double new_key = 1001.0 + i;
        double new_value = 2000.0 + i;
        sig2mod->update(new_key, new_value);
    }

    // Test lookup after retraining
    for (size_t i = 0; i < 1000; ++i)
    {
        double result = sig2mod->lookup(1001.0 + i);
        EXPECT_NEAR(result, keys.size() + i, 256.0); // Allow for larger error due to retraining
    }
}

TEST_F(Sig2ModTest, DistributionShift)
{
    auto [keys1, values1] = generate_random_data(10000, 0.0, 100000.0);
    auto [keys2, values2] = generate_random_data(10000, 2000.0, 300000.0);

    // Insert initial data
    sig2mod->insert(keys1, values1);

    // Insert data from a different distribution
    for (size_t i = 0; i < keys2.size(); ++i)
    {
        sig2mod->update(keys2[i], values2[i]);
    }

    // Test lookup for both distributions
    for (size_t i = 0; i < 1000; ++i)
    {
        double result1 = sig2mod->lookup(keys1[i]);
        EXPECT_NEAR(result1, i, 256.0);

        double result2 = sig2mod->lookup(keys2[i]);
        EXPECT_NEAR(result2, keys1.size() + i, 256.0);
    }
}

TEST_F(Sig2ModTest, PerformanceBenchmark)
{
    auto [keys, values] = generate_random_data(1000000, 0.0, 100000000.0);

    // Measure insert time
    auto start = std::chrono::high_resolution_clock::now();
    sig2mod->insert(keys, values);
    auto end = std::chrono::high_resolution_clock::now();
    auto insert_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Measure lookup time
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 10000; ++i)
    {
        size_t index = rand() % keys.size();
        volatile double result = sig2mod->lookup(keys[index]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto lookup_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Insert time for 1M keys: " << insert_duration.count() << " ms" << std::endl;
    std::cout << "Average lookup time: " << lookup_duration.count() / 10000.0 << " µs" << std::endl;

    // Add some performance expectations
    EXPECT_LT(insert_duration.count(), 10000);         // Expect insert to take less than 10 seconds
    EXPECT_LT(lookup_duration.count() / 10000.0, 100); // Expect average lookup to take less than 100 µs
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
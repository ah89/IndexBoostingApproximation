#include <gtest/gtest.h>
#include "sig2model.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <chrono>

class Sig2ModelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        sig2model = std::make_unique<Sig2Model>(
            5.0,  // error_range
            5,    // buffer_size
            10,   // batch_size
            0.01, // error_threshold
            100,  // max_iterations
            3,    // K (Gaussian components)
            3,    // N (Sigmoid capacity)
            15    // num_placeholders
        );

        // Load dataset from files
        auto [keys, values] = load_data("data/WIKI/keys_trimmed.txt", "data/WIKI/values_sync.txt");

        if (keys.empty() || values.empty()) {
            throw std::runtime_error("Failed to load test data from files.");
        }

        dataset_keys = keys;
        dataset_values = values;
    }

    std::unique_ptr<Sig2Model> sig2model;
    std::vector<double> dataset_keys;
    std::vector<double> dataset_values;

    std::pair<std::vector<double>, std::vector<double>> load_data(const std::string& keys_file, const std::string& values_file)
    {
        std::vector<double> keys, values;

        std::ifstream keys_stream(keys_file);
        std::ifstream values_stream(values_file);

        if (!keys_stream.is_open() || !values_stream.is_open()) {
            throw std::runtime_error("Error opening data files.");
        }

        std::string line;
        while (std::getline(keys_stream, line)) {
            keys.push_back(std::stod(line));
        }

        while (std::getline(values_stream, line)) {
            values.push_back(std::stod(line));
        }

        if (keys.size() != values.size()) {
            throw std::runtime_error("Keys and values file sizes do not match.");
        }

        return {keys, values};
    }
};

TEST_F(Sig2ModelTest, InsertAndLookup)
{
    // Insert data from file
    sig2model->insert(dataset_keys, dataset_values);

    // Test lookup
    for (size_t i = 0; i < dataset_keys.size(); ++i)
    {
        std::vector<double> results = sig2model->lookup(dataset_keys[i]);

        // Check if expected value is in results
        auto it = std::find(results.begin(), results.end(), dataset_values[i]);
        if (it == results.end())
        {
            throw std::runtime_error("Expected value not found in lookup results!");
        }
    }
}

TEST_F(Sig2ModelTest, Update)
{
    sig2model->insert(dataset_keys, dataset_values);

    // Perform updates and store expected values
    std::map<double, double> updated_values;
    for (size_t i = 0; i < std::min<size_t>(10, dataset_keys.size()); ++i)
    {
        double new_key = dataset_keys[i] + 0.1;
        double new_value = dataset_values[i] + 50.0;
        sig2model->update(new_key, new_value);
        updated_values[new_key] = new_value;
    }

    // Test lookup after updates
    for (const auto& [key, expected_value] : updated_values)
    {
        std::vector<double> results = sig2model->lookup(key);
        if (std::find(results.begin(), results.end(), expected_value) == results.end())
        {
            throw std::runtime_error("Updated value not found in lookup results!");
        }
    }
}

TEST_F(Sig2ModelTest, TrainAndRetrain)
{
    sig2model->insert(dataset_keys, dataset_values);

    // Perform updates to trigger retraining
    for (size_t i = 0; i < std::min<size_t>(5, dataset_keys.size()); ++i)
    {
        double new_key = dataset_keys[i] + 1.0;
        double new_value = dataset_values[i] + 100.0;
        sig2model->update(new_key, new_value);
    }

    // Test lookup after retraining
    for (size_t i = 0; i < std::min<size_t>(10, dataset_keys.size()); ++i)
    {
        double expected_value = dataset_values[i] + 100.0;
        std::vector<double> results = sig2model->lookup(dataset_keys[i] + 1.0);
        if (std::find(results.begin(), results.end(), expected_value) == results.end())
        {
            throw std::runtime_error("Retrained value not found in lookup results!");
        }
    }
}

TEST_F(Sig2ModelTest, PerformanceBenchmark)
{
    sig2model->insert(dataset_keys, dataset_values);

    // Measure insert time
    auto start = std::chrono::high_resolution_clock::now();
    sig2model->insert(dataset_keys, dataset_values);
    auto end = std::chrono::high_resolution_clock::now();
    auto insert_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Measure lookup time
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < std::min<size_t>(10000, dataset_keys.size()); ++i)
    {
        size_t index = rand() % dataset_keys.size();
        std::vector<double> results = sig2model->lookup(dataset_keys[index]);

        ASSERT_FALSE(results.empty()) << "Lookup returned an empty result for key: " << dataset_keys[index];

        volatile double first_value = results[0]; // Prevent compiler optimizations
    }
    end = std::chrono::high_resolution_clock::now();
    auto lookup_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Output performance metrics
    std::cout << "Insert time: " << insert_duration.count() << " ms" << std::endl;
    std::cout << "Average lookup time: " << lookup_duration.count() / 10000.0 << " µs" << std::endl;

    EXPECT_LT(insert_duration.count(), 10000);         // Expect insert to take < 10 seconds
    EXPECT_LT(lookup_duration.count() / 10000.0, 100); // Expect avg lookup < 100 µs
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
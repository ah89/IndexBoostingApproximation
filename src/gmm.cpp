// src/gmm.cpp
#include "gmm.h"
#include <algorithm>
#include <numeric>
#include <cmath>

GMM::GMM(int n_components, int max_iterations, double tolerance)
    : n_components_(n_components), max_iterations_(max_iterations), tolerance_(tolerance) {}

void GMM::fit(const std::vector<double>& data) {
    initialize(data);

    std::vector<std::vector<double>> responsibilities(data.size(), std::vector<double>(n_components_));
    double prev_log_likelihood = -std::numeric_limits<double>::infinity();

    for (int iteration = 0; iteration < max_iterations_; ++iteration) {
        // E-step
        expectation_step(data, responsibilities);

        // M-step
        maximization_step(data, responsibilities);

        // Check for convergence
        double log_likelihood = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            double sample_likelihood = 0;
            for (int j = 0; j < n_components_; ++j) {
                sample_likelihood += gaussians_[j].weight * gaussian_pdf(data[i], gaussians_[j].mean, gaussians_[j].variance);
            }
            log_likelihood += std::log(sample_likelihood);

            
        }

        if (std::abs(log_likelihood - prev_log_likelihood) < tolerance_) {
            break;
        }
        prev_log_likelihood = log_likelihood;
    }
}

double GMM::predict(double x) const {
    double total_probability = 0;
    for (const auto& gaussian : gaussians_) {
        total_probability += gaussian.weight * gaussian_pdf(x, gaussian.mean, gaussian.variance);
    }
    return total_probability;
}

void GMM::initialize(const std::vector<double>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);

    gaussians_.clear();
    gaussians_.reserve(n_components_);

    for (int i = 0; i < n_components_; ++i) {
        Gaussian g;
        g.mean = data[dis(gen)];
        g.variance = 1.0;
        g.weight = 1.0 / n_components_;
        gaussians_.push_back(g);
    }
}

double GMM::gaussian_pdf(double x, double mean, double variance) const {
    return 1.0 / std::sqrt(2 * M_PI * variance) * std::exp(-0.5 * std::pow(x - mean, 2) / variance);
}

void GMM::expectation_step(const std::vector<double>& data, std::vector<std::vector<double>>& responsibilities) {
    for (size_t i = 0; i < data.size(); ++i) {
        double total_responsibility = 0;
        for (int j = 0; j < n_components_; ++j) {
            responsibilities[i][j] = gaussians_[j].weight * gaussian_pdf(data[i], gaussians_[j].mean, gaussians_[j].variance);
            total_responsibility += responsibilities[i][j];
        }
        for (int j = 0; j < n_components_; ++j) {
            responsibilities[i][j] /= total_responsibility;
        }
    }
}

void GMM::maximization_step(const std::vector<double>& data, const std::vector<std::vector<double>>& responsibilities) {
    std::vector<double> N(n_components_, 0);

    for (int j = 0; j < n_components_; ++j) {
        double sum_responsibility = 0;
        double sum_responsibility_x = 0;
        double sum_responsibility_x_squared = 0;

        for (size_t i = 0; i < data.size(); ++i) {
            sum_responsibility += responsibilities[i][j];
            sum_responsibility_x += responsibilities[i][j] * data[i];
            sum_responsibility_x_squared += responsibilities[i][j] * data[i] * data[i];
        }

        N[j] = sum_responsibility;
        gaussians_[j].weight = N[j] / data.size();
        gaussians_[j].mean = sum_responsibility_x / N[j];
        gaussians_[j].variance = sum_responsibility_x_squared / N[j] - gaussians_[j].mean * gaussians_[j].mean;
    }
}
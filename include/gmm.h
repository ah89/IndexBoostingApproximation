#pragma once

#include <vector>
#include <random>

class GMM {
public:
    GMM(int n_components, int max_iterations = 100, double tolerance = 1e-3);
    void fit(const std::vector<double>& data);
    double predict(double x) const;

private:
    struct Gaussian {
        double mean;
        double variance;
        double weight;
    };

    int n_components_;
    int max_iterations_;
    double tolerance_;
    std::vector<Gaussian> gaussians_;

    void initialize(const std::vector<double>& data);
    double gaussian_pdf(double x, double mean, double variance) const;
    void expectation_step(const std::vector<double>& data, std::vector<std::vector<double>>& responsibilities);
    void maximization_step(const std::vector<double>& data, const std::vector<std::vector<double>>& responsibilities);
};

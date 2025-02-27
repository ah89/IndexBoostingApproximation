#include "neural_network.h"
#include <cmath>
#include <algorithm>

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size, double dropout_rate)
    : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size), dropout_rate_(dropout_rate) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    layers_.push_back({std::vector<std::vector<double>>(hidden_size, std::vector<double>(input_size)),
                       std::vector<double>(hidden_size),
                       std::vector<double>(hidden_size),
                       std::vector<double>(input_size)});

    layers_.push_back({std::vector<std::vector<double>>(output_size, std::vector<double>(hidden_size)),
                       std::vector<double>(output_size),
                       std::vector<double>(output_size),
                       std::vector<double>(hidden_size)});

    for (auto& layer : layers_) {
        for (auto& neuron : layer.weights) {
            for (auto& weight : neuron) {
                weight = d(gen) / std::sqrt(neuron.size());
            }
        }
        for (auto& bias : layer.biases) {
            bias = d(gen);
        }
    }
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    std::vector<double> current_input = input;

    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& layer = layers_[i];
        layer.input = current_input;
        layer.output.resize(layer.biases.size());

        for (size_t j = 0; j < layer.output.size(); ++j) {
            layer.output[j] = layer.biases[j];
            for (size_t k = 0; k < current_input.size(); ++k) {
                layer.output[j] += layer.weights[j][k] * current_input[k];
            }
            layer.output[j] = relu(layer.output[j]);

            if (i < layers_.size() - 1) {
                if (static_cast<double>(rand()) / RAND_MAX < dropout_rate_) {
                    layer.output[j] = 0;
                } else {
                    layer.output[j] /= (1 - dropout_rate_);
                }
            }
        }

        current_input = layer.output;
    }

    return current_input;
}

void NeuralNetwork::backward(const std::vector<double>& input, const std::vector<double>& gradient, double learning_rate) {
    std::vector<double> error = gradient;

    for (int i = layers_.size() - 1; i >= 0; --i) {
        auto& layer = layers_[i];

        std::vector<double> prev_error(i == 0 ? input_size_ : layers_[i-1].output.size(), 0);

        for (size_t j = 0; j < layer.weights.size(); ++j) {
            for (size_t k = 0; k < layer.weights[j].size(); ++k) {
                double delta = error[j] * (i == 0 ? input[k] : layers_[i-1].output[k]);
                layer.weights[j][k] -= learning_rate * delta;
                prev_error[k] += error[j] * layer.weights[j][k];
            }
            layer.biases[j] -= learning_rate * error[j];
        }
        if (i > 0) {
            for (size_t j = 0; j < prev_error.size(); ++j) {
                prev_error[j] *= relu_derivative(layers_[i-1].output[j]);
            }
        }
        error = prev_error;
    }
}

double NeuralNetwork::relu(double x) {
    return std::max(0.0, x);
}

double NeuralNetwork::relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& X, 
                          const std::vector<std::vector<double>>& y, 
                          int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            std::vector<double> prediction = forward(X[i]);
            backward(X[i], y[i], learning_rate);
        }
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    return forward(input);
}

std::vector<double> NNPi::forward(const std::vector<double>& input) {
    return NeuralNetwork::forward(input);
}

std::vector<std::vector<double>> NNPi::forward_detailed(const std::vector<double>& input) {
    std::vector<double> output = forward(input);
    int N = input.size();
    std::vector<std::vector<double>> result(3, std::vector<double>(N));
    for (int i = 0; i < N; ++i) {
        result[0][i] = output[i];
        result[1][i] = output[N + i];
        result[2][i] = output[2 * N + i];
    }
    return result;
}

std::vector<double> NNPhi::forward(const std::vector<double>& input) {
    return NeuralNetwork::forward(input);
}

std::vector<std::vector<double>> NNPhi::forward_detailed(const std::vector<double>& input) {
    std::vector<double> output = forward(input);
    int K = input.size();
    std::vector<std::vector<double>> result(3, std::vector<double>(K));
    for (int i = 0; i < K; ++i) {
        result[0][i] = output[i];
        result[1][i] = output[K + i];
        result[2][i] = output[2 * K + i];
    }
    return result;
}

std::vector<double> NNC::forward(const std::vector<double>& input) {
    std::vector<double> embedded = NeuralNetwork::forward(input);
    
    std::vector<double> highway(hidden_size_);
    for (int i = 0; i < hidden_size_; ++i) {
        highway[i] = relu(embedded[i]);
        embedded[i] += highway[i];
    }

    for (int i = 0; i < hidden_size_; ++i) {
        if (static_cast<double>(rand()) / RAND_MAX < dropout_rate_) {
            embedded[i] = 0;
        } else {
            embedded[i] /= (1 - dropout_rate_);
        }
    }

    std::vector<double> output(output_size_);
    for (int i = 0; i < output_size_; ++i) {
        output[i] = 0;
        for (int j = 0; j < hidden_size_; ++j) {
            output[i] += layers_.back().weights[i][j] * embedded[j];
        }
        output[i] = relu(output[i] + layers_.back().biases[i]);
    }

    return output;
}

void NNC::backward(const std::vector<double>& gradient, double learning_rate) {
    std::vector<double> error = gradient;
    
    std::vector<double> prev_error(hidden_size_, 0);
    for (int i = 0; i < output_size_; ++i) {
        for (int j = 0; j < hidden_size_; ++j) {
            double delta = error[i] * relu_derivative(layers_.back().output[i]) * layers_.back().weights[i][j];
            layers_.back().weights[i][j] -= learning_rate * delta;
            prev_error[j] += delta;
        }
        layers_.back().biases[i] -= learning_rate * error[i] * relu_derivative(layers_.back().output[i]);
    }

    for (int i = 0; i < hidden_size_; ++i) {
        if (layers_[0].output[i] != 0) {
            prev_error[i] /= (1 - dropout_rate_);
            prev_error[i] += relu_derivative(layers_[0].output[i]);
        } else {
            prev_error[i] = 0;
        }
    }

    NeuralNetwork::backward(layers_[0].input, prev_error, learning_rate);
}

ComplexNN::ComplexNN(int N, int K, int hidden_dim, int output_dim)
    : nn_pi(N), nn_phi(K), nn_c(3*N + 3*K, hidden_dim, 3*N + 3*K), N(N), K(K) {}

std::pair<std::vector<double>, std::vector<double>> ComplexNN::forward(const std::vector<double>& x_pi, const std::vector<double>& x_phi) {
    auto pi_output = nn_pi.forward(x_pi);
    auto phi_output = nn_phi.forward(x_phi);

    std::vector<double> combined_input;
    combined_input.insert(combined_input.end(), pi_output.begin(), pi_output.end());
    combined_input.insert(combined_input.end(), phi_output.begin(), phi_output.end());

    std::vector<double> c_output = nn_c.forward(combined_input);

    std::vector<double> out_pi(c_output.begin(), c_output.begin() + 3*N);
    std::vector<double> out_phi(c_output.begin() + 3*N, c_output.end());

    return {out_pi, out_phi};
}

void ComplexNN::train(const std::vector<std::vector<double>>& X_pi, 
                      const std::vector<std::vector<double>>& X_phi,
                      const std::vector<std::vector<double>>& y_pi,
                      const std::vector<std::vector<double>>& y_phi,
                      int epochs, double learning_rate) {
    if (X_pi.size() != X_phi.size() || X_pi.size() != y_pi.size() || X_pi.size() != y_phi.size()) {
        throw std::invalid_argument("All input and output datasets must have the same number of samples");
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<size_t> indices(X_pi.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), g);

        for (size_t i : indices) {
            auto [out_pi, out_phi] = forward(X_pi[i], X_phi[i]);

            double cost_pi = calculateCost(out_pi, y_pi[i]);
            std::vector<double> gradient_pi = calculateGradient(out_pi, y_pi[i]);

            nn_c.backward(gradient_pi, learning_rate);
            nn_pi.backward(X_pi[i], gradient_pi, learning_rate);

            double cost_phi = calculateCost(out_phi, y_phi[i]);
            std::vector<double> gradient_phi = calculateGradient(out_phi, y_phi[i]);

            nn_c.backward(gradient_phi, learning_rate);
            nn_phi.backward(X_phi[i], gradient_phi, learning_rate);
        }
    }
}

double ComplexNN::calculateCost(const std::vector<double>& predicted, const std::vector<double>& target) {
    double cost = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        double diff = predicted[i] - target[i];
        cost += diff * diff;
    }
    return cost / (2 * predicted.size());
}

std::vector<double> ComplexNN::calculateGradient(const std::vector<double>& predicted, const std::vector<double>& target) {
    std::vector<double> gradient(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        gradient[i] = predicted[i] - target[i];
    }
    return gradient;
}
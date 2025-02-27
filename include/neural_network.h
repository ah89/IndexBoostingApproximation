#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>

struct Layer {
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> output;
    std::vector<double> input;
};

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size, double dropout_rate);
    virtual std::vector<double> forward(const std::vector<double>& input);
    virtual void backward(const std::vector<double>& input, const std::vector<double>& gradient, double learning_rate);
    virtual void train(const std::vector<std::vector<double>>& X, 
                       const std::vector<std::vector<double>>& y, 
                       int epochs, double learning_rate);
    virtual std::vector<double> predict(const std::vector<double>& input);

protected:
    double relu(double x);
    double relu_derivative(double x);

    int input_size_;
    int hidden_size_;
    int output_size_;
    double dropout_rate_;
    std::vector<Layer> layers_;
};

class NNPi : public NeuralNetwork {
public:
    NNPi(int N) : NeuralNetwork(N, N, 3*N, 0.0) {}
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<std::vector<double>> forward_detailed(const std::vector<double>& input);
};

class NNPhi : public NeuralNetwork {
public:
    NNPhi(int K) : NeuralNetwork(K, K, 3*K, 0.0) {}
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<std::vector<double>> forward_detailed(const std::vector<double>& input);
};

class NNC : public NeuralNetwork {
public:
    NNC(int input_dim, int hidden_dim, int output_dim) 
        : NeuralNetwork(input_dim, hidden_dim, output_dim, 0.5) {}
    std::vector<double> forward(const std::vector<double>& input) override;
    void backward(const std::vector<double>& gradient, double learning_rate);
};

class ComplexNN {
public:
    ComplexNN(int N, int K, int hidden_dim, int output_dim);
    std::pair<std::vector<double>, std::vector<double>> forward(const std::vector<double>& x_pi, const std::vector<double>& x_phi);
    void train(const std::vector<std::vector<double>>& X_pi, 
               const std::vector<std::vector<double>>& X_phi,
               const std::vector<std::vector<double>>& y_pi,
               const std::vector<std::vector<double>>& y_phi,
               int epochs, double learning_rate);

private:
    NNPi nn_pi;
    NNPhi nn_phi;
    NNC nn_c;
    int N, K;

    double calculateCost(const std::vector<double>& predicted, const std::vector<double>& target);
    std::vector<double> calculateGradient(const std::vector<double>& predicted, const std::vector<double>& target);
};

#endif // NEURAL_NETWORK_H
#pragma once
#include <vector>
#include <Eigen/Dense>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& topology, double learningRate = 0.1);
    Eigen::VectorXd feedForward(const Eigen::VectorXd& input);
    void backPropagate(const Eigen::VectorXd& target);
    void train(const std::vector<Eigen::VectorXd>& inputs, const std::vector<Eigen::VectorXd>& targets, int epochs);

private:
    std::vector<int> m_topology;
    double m_learningRate;
    std::vector<Eigen::MatrixXd> m_weights;
    std::vector<Eigen::VectorXd> m_biases;
    std::vector<Eigen::VectorXd> m_layerOutputs;
    std::vector<Eigen::VectorXd> m_layerActivatedOutputs;

    static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    static double sigmoidDerivative(double x) { return x * (1.0 - x); }
};
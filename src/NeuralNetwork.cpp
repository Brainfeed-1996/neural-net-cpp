#include "NeuralNetwork.hpp"
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology, double learningRate) 
    : m_topology(topology), m_learningRate(learningRate) {
    for (size_t i = 0; i < topology.size() - 1; ++i) {
        m_weights.push_back(Eigen::MatrixXd::Random(topology[i+1], topology[i]));
        m_biases.push_back(Eigen::VectorXd::Random(topology[i+1]));
    }
}

Eigen::VectorXd NeuralNetwork::feedForward(const Eigen::VectorXd& input) {
    m_layerOutputs.clear();
    m_layerActivatedOutputs.clear();
    
    Eigen::VectorXd current = input;
    m_layerActivatedOutputs.push_back(current);

    for (size_t i = 0; i < m_weights.size(); ++i) {
        current = (m_weights[i] * current) + m_biases[i];
        m_layerOutputs.push_back(current);
        current = current.unaryExpr(&NeuralNetwork::sigmoid);
        m_layerActivatedOutputs.push_back(current);
    }
    return current;
}

void NeuralNetwork::backPropagate(const Eigen::VectorXd& target) {
    Eigen::VectorXd output = m_layerActivatedOutputs.back();
    Eigen::VectorXd error = target - output;
    
    for (int i = m_weights.size() - 1; i >= 0; --i) {
        Eigen::VectorXd gradient = m_layerActivatedOutputs[i+1].unaryExpr(&NeuralNetwork::sigmoidDerivative);
        gradient = gradient.cwiseProduct(error) * m_learningRate;

        Eigen::MatrixXd deltaWeights = gradient * m_layerActivatedOutputs[i].transpose();
        
        if (i > 0) {
            error = m_weights[i].transpose() * error;
        }

        m_weights[i] += deltaWeights;
        m_biases[i] += gradient;
    }
}

void NeuralNetwork::train(const std::vector<Eigen::VectorXd>& inputs, const std::vector<Eigen::VectorXd>& targets, int epochs) {
    for (int e = 0; e < epochs; ++e) {
        double totalError = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            Eigen::VectorXd output = feedForward(inputs[i]);
            backPropagate(targets[i]);
            totalError += (targets[i] - output).squaredNorm();
        }
        if (e % 100 == 0) std::cout << "Epoch " << e << " - Error: " << totalError << std::endl;
    }
}
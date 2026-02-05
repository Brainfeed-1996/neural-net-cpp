#include "../include/NeuralNetwork.hpp"
#include "../include/Activations.hpp"
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology, double learningRate) 
    : m_topology(topology), m_learningRate(learningRate) {
    for (size_t i = 0; i < topology.size() - 1; ++i) {
        // He Initialization for better convergence with ReLU
        m_weights.push_back(Eigen::MatrixXd::Random(topology[i+1], topology[i]) * std::sqrt(2.0/topology[i]));
        m_biases.push_back(Eigen::VectorXd::Zero(topology[i+1]));
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
        
        // Use ReLU for hidden layers, Sigmoid for output (binary classification)
        ActivationType type = (i == m_weights.size() - 1) ? ActivationType::SIGMOID : ActivationType::RELU;
        current = Activation::Apply(current, type);
        m_layerActivatedOutputs.push_back(current);
    }
    return current;
}

void NeuralNetwork::backPropagate(const Eigen::VectorXd& target) {
    Eigen::VectorXd output = m_layerActivatedOutputs.back();
    Eigen::VectorXd error = target - output;
    
    for (int i = m_weights.size() - 1; i >= 0; --i) {
        ActivationType type = (i == m_weights.size() - 1) ? ActivationType::SIGMOID : ActivationType::RELU;
        Eigen::VectorXd gradient = Activation::Derivative(m_layerOutputs[i], type); // Derivative wrt pre-activation
        
        gradient = gradient.array() * error.array(); // Element-wise mult
        gradient *= m_learningRate;

        Eigen::MatrixXd deltaWeights = gradient * m_layerActivatedOutputs[i].transpose();
        
        if (i > 0) {
            error = m_weights[i].transpose() * gradient; // Propagate error back
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
        if (e % 100 == 0) std::cout << "Epoch " << e << " - Loss: " << totalError / inputs.size() << std::endl;
    }
}
#include "../include/NeuralNetwork.hpp"
#include "../include/Activations.hpp"
#include <iostream>
#include <fstream>
#include <random>

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology, double learningRate) 
    : m_topology(topology), m_learningRate(learningRate) {
    initializeWeights();
}

void NeuralNetwork::initializeWeights() {
    for (size_t i = 0; i < m_topology.size() - 1; ++i) {
        int inputSize = m_topology[i];
        int outputSize = m_topology[i + 1];
        
        // He Initialization for ReLU
        double scale = std::sqrt(2.0 / inputSize);
        m_weights.push_back(Eigen::MatrixXd::Random(outputSize, inputSize) * scale);
        m_biases.push_back(Eigen::VectorXd::Zero(outputSize));
        
        // Initialize cache vectors
        m_layerMeans.push_back(Eigen::VectorXd::Zero(outputSize));
        m_layerVars.push_back(Eigen::VectorXd::Zero(outputSize));
    }
}

Eigen::VectorXd NeuralNetwork::feedForward(const Eigen::VectorXd& input) {
    m_layerOutputs.clear();
    m_layerActivatedOutputs.clear();
    
    Eigen::VectorXd current = input;
    m_layerActivatedOutputs.push_back(current);
    
    for (size_t i = 0; i < m_weights.size(); ++i) {
        // Linear transformation: Z = W * a + b
        current = (m_weights[i] * current) + m_biases[i];
        m_layerOutputs.push_back(current);
        
        // Apply activation
        ActivationType type = (i == m_weights.size() - 1) ? 
                              ActivationType::SIGMOID : ActivationType::RELU;
        current = Activation::Apply(current, type);
        m_layerActivatedOutputs.push_back(current);
    }
    
    return current;
}

void NeuralNetwork::backPropagate(const Eigen::VectorXd& target) {
    Eigen::VectorXd output = m_layerActivatedOutputs.back();
    Eigen::VectorXd error = target - output;
    
    for (int i = static_cast<int>(m_weights.size()) - 1; i >= 0; --i) {
        ActivationType type = (i == static_cast<int>(m_weights.size()) - 1) ? 
                              ActivationType::SIGMOID : ActivationType::RELU;
        Eigen::VectorXd gradient = Activation::Derivative(m_layerOutputs[i], type);
        
        // Element-wise multiplication with error
        gradient = gradient.array() * error.array();
        gradient *= m_learningRate;
        
        // Weight update: delta_W = gradient * a_prev^T
        Eigen::MatrixXd deltaWeights = gradient * m_layerActivatedOutputs[i].transpose();
        
        // Propagate error to previous layer
        if (i > 0) {
            error = m_weights[i].transpose() * gradient;
        }
        
        // Update weights and biases
        m_weights[i] += deltaWeights;
        m_biases[i] += gradient;
    }
}

void NeuralNetwork::train(const std::vector<Eigen::VectorXd>& inputs, 
                          const std::vector<Eigen::VectorXd>& targets, 
                          int epochs,
                          int batchSize) {
    size_t n = inputs.size();
    
    for (int e = 0; e < epochs; ++e) {
        double totalError = 0;
        
        // Mini-batch training
        for (size_t i = 0; i < n; ++i += batchSize) {
            size_t end = std::min(i + batchSize, n);
            
            // Accumulate gradients for batch
            for (size_t j = i; j < end; ++j) {
                Eigen::VectorXd output = feedForward(inputs[j]);
                backPropagate(targets[j]);
                totalError += (targets[j] - output).squaredNorm();
            }
        }
        
        if (e % 100 == 0) {
            double avgError = totalError / n;
            std::cout << "Epoch " << e << " - Loss: " << avgError << std::endl;
        }
    }
}

void NeuralNetwork::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Save topology
    size_t topologySize = m_topology.size();
    file.write(reinterpret_cast<const char*>(&topologySize), sizeof(size_t));
    for (int size : m_topology) {
        file.write(reinterpret_cast<const char*>(&size), sizeof(int));
    }
    
    // Save learning rate
    file.write(reinterpret_cast<const char*>(&m_learningRate), sizeof(double));
    
    // Save weights
    size_t numLayers = m_weights.size();
    file.write(reinterpret_cast<const char*>(&numLayers), sizeof(size_t));
    
    for (const auto& w : m_weights) {
        int rows = w.rows();
        int cols = w.cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(w.data()), rows * cols * sizeof(double));
    }
    
    // Save biases
    for (const auto& b : m_biases) {
        int size = b.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(int));
        file.write(reinterpret_cast<const char*>(b.data()), size * sizeof(double));
    }
    
    file.close();
    std::cout << "Model saved to: " << filename << std::endl;
}

bool NeuralNetwork::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    
    try {
        // Load topology
        size_t topologySize;
        file.read(reinterpret_cast<char*>(&topologySize), sizeof(size_t));
        m_topology.clear();
        for (size_t i = 0; i < topologySize; ++i) {
            int size;
            file.read(reinterpret_cast<char*>(&size), sizeof(int));
            m_topology.push_back(size);
        }
        
        // Load learning rate
        file.read(reinterpret_cast<char*>(&m_learningRate), sizeof(double));
        
        // Load weights
        size_t numLayers;
        file.read(reinterpret_cast<char*>(&numLayers), sizeof(size_t));
        m_weights.clear();
        
        for (size_t i = 0; i < numLayers; ++i) {
            int rows, cols;
            file.read(reinterpret_cast<char*>(&rows), sizeof(int));
            file.read(reinterpret_cast<char*>(&cols), sizeof(int));
            Eigen::MatrixXd w(rows, cols);
            file.read(reinterpret_cast<char*>(w.data()), rows * cols * sizeof(double));
            m_weights.push_back(w);
        }
        
        // Load biases
        m_biases.clear();
        for (size_t i = 0; i < numLayers; ++i) {
            int size;
            file.read(reinterpret_cast<char*>(&size), sizeof(int));
            Eigen::VectorXd b(size);
            file.read(reinterpret_cast<char*>(b.data()), size * sizeof(double));
            m_biases.push_back(b);
        }
        
        file.close();
        std::cout << "Model loaded from: " << filename << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

double NeuralNetwork::calculateLoss(const std::vector<Eigen::VectorXd>& inputs,
                                    const std::vector<Eigen::VectorXd>& targets) const {
    double totalError = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        // Clone network for inference (non-const method limitation)
        // For now, use MSE directly
        Eigen::VectorXd output = const_cast<NeuralNetwork*>(this)->feedForward(inputs[i]);
        totalError += (targets[i] - output).squaredNorm();
    }
    return totalError / inputs.size();
}

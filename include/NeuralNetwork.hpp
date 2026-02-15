#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>

enum class ActivationType {
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX,
    LEAKY_RELU,
    ELU
};

class Activation {
public:
    static Eigen::VectorXd Apply(const Eigen::VectorXd& x, ActivationType type);
    static Eigen::VectorXd Derivative(const Eigen::VectorXd& x, ActivationType type);
    static std::string toString(ActivationType type);
};

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& topology, double learningRate = 0.1);
    
    // Forward/backward pass
    Eigen::VectorXd feedForward(const Eigen::VectorXd& input);
    void backPropagate(const Eigen::VectorXd& target);
    void train(const std::vector<Eigen::VectorXd>& inputs, 
               const std::vector<Eigen::VectorXd>& targets, 
               int epochs,
               int batchSize = 1);
    
    // Serialization
    void save(const std::string& filename) const;
    bool load(const std::string& filename);
    
    // Getters/Setters
    void setLearningRate(double lr) { m_learningRate = lr; }
    double getLearningRate() const { return m_learningRate; }
    std::vector<int> getTopology() const { return m_topology; }
    
    // Utility
    double calculateLoss(const std::vector<Eigen::VectorXd>& inputs,
                         const std::vector<Eigen::VectorXd>& targets) const;
    
private:
    std::vector<int> m_topology;
    double m_learningRate;
    std::vector<Eigen::MatrixXd> m_weights;
    std::vector<Eigen::VectorXd> m_biases;
    std::vector<Eigen::VectorXd> m_layerOutputs;
    std::vector<Eigen::VectorXd> m_layerActivatedOutputs;
    
    // Cache for batch normalization (future feature)
    std::vector<Eigen::VectorXd> m_layerMeans;
    std::vector<Eigen::VectorXd> m_layerVars;
    
    void initializeWeights();
    ActivationType getOutputActivation() const { return ActivationType::SIGMOID; }
};

#pragma once
#include <cmath>
#include <string>
#include <Eigen/Dense>

enum class ActivationType {
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX
};

class Activation {
public:
    static Eigen::VectorXd Apply(const Eigen::VectorXd& x, ActivationType type) {
        switch(type) {
            case ActivationType::SIGMOID: return x.unaryExpr([](double v){ return 1.0 / (1.0 + std::exp(-v)); });
            case ActivationType::RELU:    return x.unaryExpr([](double v){ return v > 0.0 ? v : 0.0; });
            case ActivationType::TANH:    return x.unaryExpr([](double v){ return std::tanh(v); });
            case ActivationType::SOFTMAX: {
                Eigen::VectorXd exp = x.array().exp();
                return exp / exp.sum();
            }
        }
        return x;
    }

    static Eigen::VectorXd Derivative(const Eigen::VectorXd& x, ActivationType type) {
        switch(type) {
            case ActivationType::SIGMOID: {
                Eigen::VectorXd s = Apply(x, ActivationType::SIGMOID);
                return s.array() * (1.0 - s.array());
            }
            case ActivationType::RELU: return x.unaryExpr([](double v){ return v > 0.0 ? 1.0 : 0.0; });
            case ActivationType::TANH: {
                Eigen::VectorXd t = Apply(x, ActivationType::TANH);
                return 1.0 - t.array().square();
            }
            case ActivationType::SOFTMAX: return Eigen::VectorXd::Ones(x.size()); // Simplified for CrossEntropy
        }
        return Eigen::VectorXd::Ones(x.size());
    }
};
#include "include/NeuralNetwork.hpp"
#include <iostream>

int main() {
    // XOR Problem
    std::vector<int> topology = {2, 4, 1};
    NeuralNetwork nn(topology, 0.5);

    std::vector<Eigen::VectorXd> inputs = {
        (Eigen::VectorXd(2) << 0, 0).finished(),
        (Eigen::VectorXd(2) << 0, 1).finished(),
        (Eigen::VectorXd(2) << 1, 0).finished(),
        (Eigen::VectorXd(2) << 1, 1).finished()
    };

    std::vector<Eigen::VectorXd> targets = {
        (Eigen::VectorXd(1) << 0).finished(),
        (Eigen::VectorXd(1) << 1).finished(),
        (Eigen::VectorXd(1) << 1).finished(),
        (Eigen::VectorXd(1) << 0).finished()
    };

    std::cout << "Training for XOR..." << std::endl;
    nn.train(inputs, targets, 2000);

    std::cout << "\nTesting XOR:" << std::endl;
    for (const auto& in : inputs) {
        std::cout << in.transpose() << " => " << nn.feedForward(in).transpose() << std::endl;
    }

    return 0;
}
# Neural Net C++ (Eigen-powered)

A high-performance, matrix-based Neural Network implementation in C++ using the **Eigen** library for linear algebra.

## 🧠 Overview

This is a modular, extensible neural network library supporting multiple activation functions, optimizers, and training modes.

## 🚀 Quick Start

```bash
mkdir build && cd build
cmake .. && make
./NeuralNet
```

## 📦 Dependencies

- **C++17** or higher
- **Eigen 3.4+** (header-only linear algebra library)

Install on Ubuntu:
```bash
sudo apt-get install libeigen3-dev
```

Install on macOS:
```bash
brew install eigen
```

## 💻 Usage

### Basic XOR Example

```cpp
#include "include/NeuralNetwork.hpp"

int main() {
    std::vector<int> topology = {2, 4, 1};
    NeuralNetwork nn(topology, 0.1);
    
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
    
    nn.train(inputs, targets, 1000);
    return 0;
}
```

### Advanced: Custom Architecture

```cpp
// 4-layer network: 784 inputs -> 128 hidden -> 64 hidden -> 10 outputs
std::vector<int> topology = {784, 128, 64, 10};
NeuralNetwork nn(topology, 0.01);

// Use different learning rate per layer
nn.setLearningRate(0.01);

// Save/load models
nn.save("model.bin");
nn.load("model.bin");
```

## 📖 Documentation

- [Architecture](ARCHITECTURE.md) - System design and components
- [Features](FEATURES.md) - Supported activations, optimizers, features
- [Usage](USAGE.md) - Detailed usage guide with examples

## 🔧 Building

```bash
# Using CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Run tests
./test_nn
```

## 📝 License

MIT License

# Usage Guide - Neural Net C++

## Installation

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libeigen3-dev
```

### macOS
```bash
brew install eigen cmake
```

### Windows
Download Eigen from http://eigen.tuxfamily.org/ and extract to include path.

## Building

```bash
git clone https://github.com/Brainfeed-1996/neural-net-cpp.git
cd neural-net-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

## Basic Usage

### 1. Include Headers

```cpp
#include "include/NeuralNetwork.hpp"
#include "include/Activations.hpp"
```

### 2. Create Network

```cpp
// Input: 2, Hidden: 4, Output: 1
std::vector<int> topology = {2, 4, 1};
NeuralNetwork nn(topology, 0.1);  // 0.1 = learning rate
```

### 3. Prepare Data

```cpp
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
```

### 4. Train

```cpp
nn.train(inputs, targets, 10000);
```

### 5. Predict

```cpp
Eigen::VectorXd test_input = (Eigen::VectorXd(2) << 0, 1).finished();
Eigen::VectorXd prediction = nn.feedForward(test_input);
std::cout << "Prediction: " << prediction.transpose() << std::endl;
```

## Complete Example: XOR

```cpp
#include "include/NeuralNetwork.hpp"
#include <iostream>

int main() {
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
        auto output = nn.feedForward(in);
        std::cout << in.transpose() << " => " << output.transpose() 
                  << " (rounded: " << (output[0] > 0.5 ? 1 : 0) << ")" << std::endl;
    }

    return 0;
}
```

## Tips

- **Learning Rate**: Start with 0.01-0.1, adjust based on convergence
- **Hidden Layers**: More neurons = more capacity, but risk overfitting
- **Epochs**: Monitor loss to detect convergence
- **Initialization**: He init works best with ReLU

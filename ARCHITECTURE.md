# Architecture - Neural Net C++

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Neural Network Library                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    NeuralNetwork                         │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  - topology: std::vector<int>                          │    │
│  │  - weights: std::vector<Eigen::MatrixXd>               │    │
│  │  - biases: std::vector<Eigen::VectorXd>                │    │
│  │  - layerOutputs: std::vector<Eigen::VectorXd>         │    │
│  │  - layerActivatedOutputs: std::vector<Eigen::VectorXd>│    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      Activation                           │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  - Apply(): Forward pass activation                     │    │
│  │  - Derivative(): Backward pass derivative               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### NeuralNetwork Class

**Public Methods:**
```cpp
NeuralNetwork(const std::vector<int>& topology, double learningRate = 0.1)
Eigen::VectorXd feedForward(const Eigen::VectorXd& input)
void backPropagate(const Eigen::VectorXd& target)
void train(const inputs, targets, epochs)
void save(const std::string& filename)
bool load(const std::string& filename)
```

**Private Members:**
```cpp
std::vector<int> m_topology           // Layer sizes [input, hidden1, ..., output]
double m_learningRate                   // SGD learning rate
std::vector<Eigen::MatrixXd> m_weights // Weight matrices W[l]
std::vector<Eigen::VectorXd> m_biases  // Bias vectors b[l]
```

### Activation Module

**Supported Activations:**
| Type | Forward f(x) | Derivative f'(x) | Use Case |
|------|-------------|-------------------|----------|
| SIGMOID | 1/(1+e⁻ˣ) | σ(x)(1-σ(x)) | Binary output |
| RELU | max(0,x) | 1 if x>0 else 0 | Hidden layers |
| TANH | tanh(x) | 1-tanh²(x) | Hidden layers |
| SOFTMAX | eˣ/Σeˣ | Simplified | Multi-class output |

## Data Flow

```
Input Vector
    │
    ▼
┌─────────────────┐
│ Z[l] = W[l]·a[l-1] + b[l]  │  (Linear transformation)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ a[l] = f(Z[l])  │  (Activation function)
└─────────────────┘
    │
    ▼
    ... (repeat for each layer)
    │
    ▼
Output Vector
```

## Training Loop

```
for epoch in epochs:
    for each (input, target) in dataset:
        output = feedForward(input)
        error = target - output
        backPropagate(error)
    if epoch % 100 == 0:
        print_loss()
```

## File Structure

```
neural-net-cpp/
├── include/
│   ├── NeuralNetwork.hpp
│   ├── Activations.hpp
│   └── core/
│       └── memory_pool.h
├── src/
│   └── NeuralNetwork.cpp
├── main.cpp
├── README.md
├── ARCHITECTURE.md
├── FEATURES.md
└── USAGE.md
```

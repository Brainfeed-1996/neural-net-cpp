# Features - Neural Net C++

## Core Features

### Multiple Activation Functions
- **ReLU**: Fast, avoids vanishing gradient
- **Sigmoid**: Binary classification output
- **Tanh**: Zero-centered, better gradients
- **Softmax**: Multi-class classification

### Training Modes
- **Stochastic Gradient Descent (SGD)**: Per-sample updates
- **Mini-Batch**: Batch-based updates (WIP)
- **Momentum**: Accelerated SGD (WIP)

### Model Persistence
- **Save/Load**: Binary format for weights and biases
- **Serialization**: JSON export for inspection

## Technical Features

### Memory Optimization
- **Eigen Library**: BLAS/LAPACK backend
- **In-place operations**: Minimize memory allocations
- **Memory Pool**: Pre-allocated buffers (optional)

### Numerical Stability
- **He Initialization**: Optimal for ReLU
- **Xavier Initialization**: Optional for tanh/sigmoid
- **Gradient Clipping**: Prevent exploding gradients

## API Reference

### NeuralNetwork

```cpp
// Constructor
NeuralNetwork(std::vector<int> topology, double learningRate = 0.1)

// Training
void train(inputs, targets, epochs)

// Inference
Eigen::VectorXd predict(const Eigen::VectorXd& input)

// Serialization
void save(const std::string& path)
bool load(const std::string& path)

// Getters
double getLearningRate() const
std::vector<int> getTopology() const
```

### Activation

```cpp
// Apply activation
Eigen::VectorXd Apply(const Eigen::VectorXd& x, ActivationType type)

// Calculate derivative
Eigen::VectorXd Derivative(const Eigen::VectorXd& x, ActivationType type)
```

## Performance

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Feed Forward | O(n×m) | n=input size, m=output size |
| Backpropagation | O(n×m) | Same as forward |
| Memory | O(Σw×h) | Sum of all weight matrices |

## Extensibility

### Adding New Activations

```cpp
enum class ActivationType {
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX,
    LEAKY_RELU  // Add here
};

// In Activation::Apply
case ActivationType::LEAKY_RELU: 
    return x.unaryExpr([](double v){ return v > 0.0 ? v : 0.01 * v; });
```

### Adding Optimizers

```cpp
class Optimizer {
    virtual void update(weights, gradients, learningRate) = 0;
};

// Implement SGD, Adam, RMSprop, etc.
```

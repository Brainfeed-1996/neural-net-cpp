# Neural Net C++ (Eigen-powered)

A high-performance, matrix-based Neural Network implementation in C++ using the **Eigen** library.

## 🧠 Architecture
The network uses a vectorized approach to handle layers and activations, significantly improving training speed compared to traditional iterative methods.

```mermaid
graph LR
    Input[Input Vector] --> W1[Weight Matrix 1]
    W1 --> H1[Hidden Layer (ReLU)]
    H1 --> S1[ReLU Activation]
    S1 --> W2[Weight Matrix 2]
    W2 --> Output[Output Vector (Sigmoid)]
    Output --> S2[Sigmoid Activation]
```

## 🔬 Mathematical Foundation
- **Activations**: 
  - Hidden Layers: **ReLU** $f(x) = max(0, x)$ (Avoids vanishing gradient)
  - Output Layer: **Sigmoid** $\sigma(x) = \frac{1}{1 + e^{-x}}$
- **Initialization**: **He Initialization** $\mathcal{N}(0, \sqrt{\frac{2}{n_{in}}})$
- **Optimization**: Stochastic Gradient Descent (SGD)
- **Loss**: Mean Squared Error (MSE)

## 🛠️ Prerequisites
- C++17 compiler
- Eigen 3.4+

## 🚀 Usage
```bash
mkdir build && cd build
cmake ..
make
./NeuralNet
```